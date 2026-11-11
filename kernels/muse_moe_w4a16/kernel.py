# okc: problem=muse_moe_w4a16
"""Routed experts with NVFP4 weights (Muse Spark MoE block, 32 local experts of 256).

What it computes, per token t and each of its top-8 routes (expert e, weight g) that falls on
one of this GPU's experts:

    h = x[t] @ dequant(w13[e]).T;  gate, up = split(h)
    y = (silu(gate) * up) @ dequant(w2[e]).T
    out[t] += g * rmsnorm(y * (1 + post_norm_w))

Kernels, in launch order (one kernel_fn call):

1. Route prep: moe_route (CUDA, one CTA) for 64 < pairs <= 4096, _route_prep (Triton) otherwise.
   Groups this GPU's (token, expert) pairs by expert, cuts each expert into blocks of <= 16
   tokens, and zeroes the GEMMs' work-item counters. Replaces expert_map + moe_align_block_size.
2. moe_gemm1 (CUDA): per block and 16 columns of I, x . Wgate^T and x . Wup^T. Without split-K
   the epilogue applies the expert's global scale and writes silu(gate) * up in BF16; with
   split-K it writes FP32 gate/up slices.
3. _swiglu_reduce (Triton, split-K only): sums the slices and applies silu(gate) * up.
4. moe_gemm2 (CUDA): per block and 32 rows of H, inter . W2^T; the epilogue applies the
   global scale and (1 + post_norm_w) and writes y (BF16, or FP32 slices with split-K).
5. _combine (Triton, one CTA per token): RMSNorm of each of the token's y rows (summing the
   K slices first), weighted sum.

Dispatch by token count bs (pairs = 8 bs), see kernel_fn and _split_k:
- K slices: 4 for bs <= 2, 2 for bs <= 64, 1 above (then step 3 is skipped).
- Programmatic dependent launch (PDL) for bs <= PDL_MAX_BS = 16.
- moe_gemm*_exact instead of moe_gemm* when prepare() finds a scale tensor with a zero,
  subnormal, negative or NaN E4M3 scale (_needs_exact); the judge's weights take the fast path.

Where to start: the GEMMs are gemm() and tile_loop() in CUDA_SRC; read them with the
dequantization and layout notes below, then pack_codes/pack_scales, then kernel_fn.

GEMMs. Each warp is independent and persistent: it takes a work item (block, columns, K
slice), streams two 16-row tiles of weights from global memory with one 16-byte load per
lane per 64-k chunk, decodes them in registers and runs mma.sync m16n8k16 with the
weights as A ("swap AB": the weight rows fill the MMA's M side, the few tokens its N side).
No shared memory, no barriers. A warp's first item is fixed; each further one comes from an
atomic counter, so warps that finish early take more work.

Dequantization is exact. An E2M1 code s|e1 e0|m written into a BF16 as sign bit 15 and
e1 e0 m at bits 8..6 reads as e2m1 * 2^-126 (a subnormal for e = 0, which is what E2M1
means by e = 0); one bf16x2 multiply by block_scale * 2^56 gives e2m1 * block_scale * 2^-70,
exact (2 + 4 significant bits fit BF16's 8), and the epilogue multiplies the FP32
accumulator by global_scale * 2^70. The E4M3 block scale becomes block_scale * 2^56 with
one byte permute and one LOP3 when every scale of the tensor is positive and normal:
prepare() stores those bytes re-encoded (recode_scales, a bijection) so that the BF16
pair is (replicated byte & SCALE_FAST_MASK) | SCALE_FAST_BIAS. Otherwise the kernels read
the original E4M3 bytes and use two exact FP32 multiplies (_needs_exact decides).

The decode's bit operations are generated in Python: CODE_SIGN_BIT / CODE_MAG_LOW say where
pack_codes puts each code's bits, and _decode_source() turns them into the body of the CUDA
decode() (filled into CUDA_SRC at @DECODE@, with the scale constants at @SCALE_FAST_*@).

Layout (pack_codes, pack_scales): prepare() reorders bits, never values. Inside
each 64-k chunk the MMA fragment position (step kk, lane c, half h, b) reads the real k
16c + 4kk + 2h + b (the sum over k doesn't care about order), so a lane's weights for a
chunk are one 16-byte load, its 16 activations per token are contiguous, and its 16 k
fall in one E4M3 block: one scale per lane, row and chunk.

Split-K (bs <= 64): each work item covers 1/4 (bs <= 2) or 1/2 (bs <= 64) of K and the
slices are summed in FP32 (_swiglu_reduce for W13, _combine for W2). With few tokens there
are too few items to fill the GPU; with more, finer items shorten the idle tail of the
last round of the persistent grid.

Few tokens (bs <= PDL_MAX_BS): the five kernels are chained by
programmatic dependent launch. Each kernel lets its dependent launch at once
(griddepcontrol.launch_dependents) and every dependent waits for its primary before reading
what it wrote (griddepcontrol.wait), so a kernel's launch and prologue overlap the previous
kernel's tail instead of following it. The CUDA kernels are launched with cuLaunchKernelEx and
the programmatic-serialization attribute (_launch_pdl), the Triton ones with launch_pdl.
Without the attribute (bs > PDL_MAX_BS) both instructions are no-ops.
"""

from typing import TypedDict

import ctypes

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait


# ---- Harness interface ----------------------------------------------------------------------

class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


# ---- GEMM geometry and the bit layout of the packed weights ----------------------------------

# 128-thread CTAs, 4 per SM (16 warps): on H200 that beats 3 by ~3.5% (more warps for the
# near-uniform work items). The GEMMs' __launch_bounds__(128, 4) holds ptxas to 128
# registers so the 4 CTAs fit; the fast-path kernels don't spill there (the _exact ones,
# for subnormal/zero/negative scales, spill ~100 B: tools/compile_cuda_sm90.py).
CUDA_CTAS_PER_SM = 4
# CHUNK: k values per MMA chunk (4 k16 steps); TILE: weight rows per A tile; GROUP: chunks per
# scale load (one 16-byte scale load covers 8 chunks of a lane's two rows).
CHUNK, TILE, GROUP = 64, 16, 8

# Where register j's code sits inside each 16-bit half of a word: sign bit and lowest of the
# three exponent/mantissa bits. Chosen so the decode is cheap (tools/search_imad_layout.py
# found it as the only layout at the minimum): j = 0 needs one mask, j = 1 a shift and a mask,
# j = 2 one IMAD between two masks, j = 3 two shifts and two masks. 10 integer ops per word
# (the unpermuted layout takes 15).
CODE_SIGN_BIT = (15, 12, 14, 13)
CODE_MAG_LOW = (6, 3, 0, 9)

# scale_of's fast path: the bf16x2 pair (s * 2^56, s * 2^56) of a recode_scales byte X is
# (X replicated into all four bytes & SCALE_FAST_MASK) | SCALE_FAST_BIAS.
SCALE_FAST_MASK, SCALE_FAST_BIAS = 0x07F007F0, 0x58005800


# ---- Generator of the CUDA decode() -----------------------------------------------------------
# Pure Python, run once at import. The generated line for register j is one of:
#   shifts equal:     (q << a) & 0x81C081C0
#   one IMAD:         ((q & src_mask) * (2^a + 2^b)) & 0x81C081C0
#   general:          ((q << a) & 0x80008000) | ((q << b or q >> -b) & 0x01C001C0)
# followed by hmul2(bits, scale pair of row g or g + 8). _decode_source() returns the four lines.

def _decode_shifts(j: int) -> tuple[int, int]:
    """Left shift that brings register j's sign bit to 15, and shift that brings its
    exponent/mantissa to 8..6 (negative = right). Left shifts move low-half bits only into
    high-half positions outside the masks; a right shift of at most 7 moves high-half bits
    only into low-half bits 9..15, outside the magnitude mask."""
    return 15 - CODE_SIGN_BIT[j], 6 - CODE_MAG_LOW[j]


def _imad_form(j: int) -> tuple[int, int] | None:
    """(source mask, multiplier) when register j's two left shifts can be one multiply:
    (q & mask) * (2^a + 2^b) sums two copies that never share a bit (so no carries), and
    neither copy puts a stray bit on a target position. None otherwise."""
    a, b = _decode_shifts(j)
    if a == b or a < 0 or b < 0:
        return None
    s, m = CODE_SIGN_BIT[j], CODE_MAG_LOW[j]
    src = [s, m, m + 1, m + 2]
    src += [p + 16 for p in src]
    copy_a, copy_b = {p + a for p in src}, {p + b for p in src}
    target = {15, 8, 7, 6, 31, 24, 23, 22}
    want = {s + a, s + 16 + a} | {p + b for p in src if p % 16 != s}
    if copy_a & copy_b or ({p for p in (copy_a | copy_b) - want if p < 32} & target):
        return None
    return sum(1 << p for p in src), (1 << a) + (1 << b)


def _shift(x: str, n: int) -> str:
    return x if n == 0 else f"({x} << {n})" if n > 0 else f"({x} >> {-n})"


def _decode_source() -> str:
    """The body of the CUDA decode(): register j = sign at 15/31 | e1 e0 m at 8..6/24..22."""
    lines = []
    for j in range(4):
        a, b = _decode_shifts(j)
        imad = _imad_form(j)
        if a == b:
            bits = f"{_shift('q', a)} & 0x81C081C0u"
        elif imad:
            bits = f"((q & 0x{imad[0]:08X}u) * {imad[1]}u) & 0x81C081C0u"
        else:
            bits = f"({_shift('q', a)} & 0x80008000u) | ({_shift('q', b)} & 0x01C001C0u)"
        lines.append(f"  r{j} = hmul2({bits}, {'pg' if j % 2 == 0 else 'pg8'});")
    return "\n".join(lines)


# ---- CUDA source: decode, GEMMs (moe_gemm1/2) and route prep (moe_route) ------------------------
# A template: @DECODE@, @SCALE_FAST_MASK@ and @SCALE_FAST_BIAS@ are filled right after it. The
# source includes no headers: uint4 and the intrinsics are NVRTC built-ins, and BF16 values are
# 16-bit patterns handled through inline PTX.

CUDA_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;

// Programmatic dependent launch (sm_90). Both are no-ops when the kernel was launched
// without the programmatic-serialization attribute.
#define GDC_LAUNCH() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
#define GDC_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")

// ---- loads and BF16 PTX wrappers ----
// ldg_stream for the weights, each byte read once (no L1 allocation); ldg_keep for the
// activations, which the CTA's other warps read again from L1, and for the small scale loads.

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ uint4 ldg_keep(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ u32 hmul2(u32 a, u32 b) {
  u32 d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ void mma(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ u16 f2bf(float v) {
  u16 r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(v));
  return r;
}

// ---- dequantization ----

// (scale, scale) as bf16x2, equal to E4M3 scale * 2^56, for byte k (0..3) of word w.
// Fast path: prepare() (recode_scales) re-encoded each positive normal E4M3 b = e3..e0 m2..m0
// as the byte X = e0 m2 m1 m0 0 e3 e2 e1 (a bijection, still 8 bits). The BF16 of scale * 2^56 is
// 0x5800 | e << 7 | m << 4: high byte 0x58 | e3 e2 e1 = 0x58 | (X & 0x07), low byte
// e0 m2 m1 m0 0000 = X & 0xF0. One byte permute replicates X into all four bytes and one
// LOP3 masks and adds the constant. EXACT path (zero, subnormal, negative or NaN scales
// somewhere in the tensor): the original E4M3 byte and two exact FP32 multiplies (an E4M3
// NaN comes out as +-480; see _needs_exact).
template <bool EXACT>
__device__ __forceinline__ u32 scale_of(u32 w, int k, u32 bias) {
  if (EXACT) {
    const u32 b = (w >> (8 * k)) & 0xFFu;
    float f = __int_as_float(((b & 0x7Fu) << 20) | ((b & 0x80u) << 24));
    f = f * 1.329227995784916e36f;   // 2^120
    f = f * 7.205759403792794e16f;   // 2^56
    const u32 h = __float_as_uint(f) >> 16;
    return h | (h << 16);
  } else {
    const u32 x4 = __byte_perm(w, 0, 0x1111u * (u32)k);
    return (x4 & @SCALE_FAST_MASK@u) | bias;  // bias in a register: one LOP3, not two
  }
}

// Decode word q (8 E2M1 codes, pack_codes layout) into the four A registers and scale them.
// Generated from CODE_SIGN_BIT / CODE_MAG_LOW by _decode_source().
__device__ __forceinline__ void decode(u32 q, u32 pg, u32 pg8, u32& r0, u32& r1, u32& r2, u32& r3) {
@DECODE@
}

__device__ __forceinline__ u32 word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// ---- GEMM work item and persistent loop ----

// One work item: NT n8 token tiles (NT = 2 when the block has more than 8 tokens), two A
// tiles (t0, t1), chunks [ch0, ch1). acc[tile][token tile][mma accumulator].
template <int NT, bool EXACT>
__device__ __forceinline__ void tile_loop(
    const uint4* __restrict__ wt0, const uint4* __restrict__ wt1,
    const uint4* __restrict__ st0, const uint4* __restrict__ st1,
    const uint4* x0, const uint4* x1, int lane, int ch0, int ch1, float acc[2][2][4]) {
  const int c = lane & 3;
  // SCALE_FAST_BIAS, kept in a register: ch1 >> 30 is always 0 (ch1 <= 64), but ptxas can't fold it.
  const u32 sbias = @SCALE_FAST_BIAS@u | ((u32)ch1 >> 30);
  for (int grp = ch0 >> 3; grp < (ch1 >> 3); ++grp) {
    // One 64-bit base per group and tile: the loads below take immediate offsets instead of
    // a sign extension, a 64-bit add and a LEA each.
    const uint4* const w0g = wt0 + (long long)grp * 256;
    const uint4* const w1g = wt1 + (long long)grp * 256;
    const uint4 s0 = ldg_keep(st0 + (long long)grp * 32);
    const uint4 s1 = ldg_keep(st1 + (long long)grp * 32);
    uint4 a0[8], a1[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      a0[u] = ldg_stream(w0g + u * 32);
      a1[u] = ldg_stream(w1g + u * 32);
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int ch = grp * 8 + u;
      uint4 b[NT][2];
      b[0][0] = ldg_keep(x0 + ch * 8 + 2 * c);
      b[0][1] = ldg_keep(x0 + ch * 8 + 2 * c + 1);
      if (NT > 1) {
        b[NT - 1][0] = ldg_keep(x1 + ch * 8 + 2 * c);
        b[NT - 1][1] = ldg_keep(x1 + ch * 8 + 2 * c + 1);
      }
      // Byte 2u of the group's scale load is row g's scale for chunk u, byte 2u + 1 row g + 8's.
      const u32 p0g = scale_of<EXACT>(word(s0, u >> 1), 2 * (u & 1), sbias);
      const u32 p0g8 = scale_of<EXACT>(word(s0, u >> 1), 2 * (u & 1) + 1, sbias);
      const u32 p1g = scale_of<EXACT>(word(s1, u >> 1), 2 * (u & 1), sbias);
      const u32 p1g8 = scale_of<EXACT>(word(s1, u >> 1), 2 * (u & 1) + 1, sbias);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        u32 r0, r1, r2, r3;
        decode(word(a0[u], kk), p0g, p0g8, r0, r1, r2, r3);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
          mma(acc[0][nt], r0, r1, r2, r3, word(b[nt][kk >> 1], (kk & 1) * 2), word(b[nt][kk >> 1], (kk & 1) * 2 + 1));
        decode(word(a1[u], kk), p1g, p1g8, r0, r1, r2, r3);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
          mma(acc[1][nt], r0, r1, r2, r3, word(b[nt][kk >> 1], (kk & 1) * 2), word(b[nt][kk >> 1], (kk & 1) * 2 + 1));
      }
    }
  }
}

// GATED (GEMM1): A tiles = gate tile wt and up tile N/16 + wt, output 16 columns of inter.
// Not GATED (GEMM2): A tiles = 2 wt and 2 wt + 1, output 32 rows of y.
template <bool GATED, bool EXACT>
__device__ void gemm(const u16* __restrict__ x, const int* __restrict__ tok,
                     const int* __restrict__ blk_exp, const int* __restrict__ blk_row,
                     const int* __restrict__ blk_cnt, const int* __restrict__ nblk_ptr,
                     const uint4* __restrict__ w, const uint4* __restrict__ sf,
                     const float* __restrict__ gs, const u16* __restrict__ nw,
                     void* __restrict__ out, int R, int local_start, int K, int N, int split) {
  GDC_LAUNCH();  // the next kernel may launch now; it waits for this grid before reading out
  GDC_WAIT();    // the previous kernel (route prep, or the SwiGLU reduce) has finished
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, c = lane & 3;
  const int warps = (gridDim.x * blockDim.x) >> 5;
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int KC = K >> 6;
  const int tiles = (GATED ? 2 * N : N) >> 4;        // 16-row A tiles per expert
  const int nwt = GATED ? (N >> 4) : (N >> 5);       // warp tiles
  const int nblk = *nblk_ptr;
  const int chs = KC / split;
  const int items = nblk * nwt * split;  // < 2^31: nblk <= pairs / 16 + 32, nwt <= 512, split <= 4
  // Dynamic item queue: a warp's first item is gw; each further one is warps + the
  // next value of a counter in nblk (zeroed by moe_route / _route_prep in every call), so warps
  // that finish early take more and the last round isn't left to a few slow ones.
  int* const counter = const_cast<int*>(nblk_ptr) + (GATED ? 2 : 3);
  for (int item = gw; item < items;) {
    // Consecutive items (the 4 warps of a CTA) take consecutive weight tiles of the same
    // token block: the block's activations come from L2 once per CTA and hit L1 for the
    // other warps (a block-fastest order would make each warp re-read them from L2).
    const int wtile = item % nwt;
    const int b = (item / nwt) % nblk;
    const int sk = item / (nblk * nwt);
    const long long ge = blk_exp[b] + local_start;
    const int row0 = blk_row[b];
    const int cnt = blk_cnt[b];
    const int t0 = GATED ? wtile : 2 * wtile;
    const int t1 = GATED ? (N >> 4) + wtile : 2 * wtile + 1;
    const uint4* wt0 = w + ((ge * tiles + t0) * KC) * 32 + lane;
    const uint4* wt1 = w + ((ge * tiles + t1) * KC) * 32 + lane;
    const uint4* st0 = sf + ((ge * tiles + t0) * (KC >> 3)) * 32 + lane;
    const uint4* st1 = sf + ((ge * tiles + t1) * (KC >> 3)) * 32 + lane;
    // Activation rows of this lane's two tokens (g and 8 + g of the block). Padding tokens
    // read a valid row; their columns are never stored.
    const int n0 = g < cnt ? g : 0, n1 = 8 + g < cnt ? 8 + g : 0;
    const int row_a = GATED ? tok[row0 + n0] : row0 + n0;
    const int row_b = GATED ? tok[row0 + n1] : row0 + n1;
    const uint4* x0 = reinterpret_cast<const uint4*>(x + (long long)row_a * K);
    const uint4* x1 = reinterpret_cast<const uint4*>(x + (long long)row_b * K);
    float acc[2][2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
    if (cnt > 8)
      tile_loop<2, EXACT>(wt0, wt1, st0, st1, x0, x1, lane, sk * chs, (sk + 1) * chs, acc);
    else
      tile_loop<1, EXACT>(wt0, wt1, st0, st1, x0, x1, lane, sk * chs, (sk + 1) * chs, acc);
    // Claim the next item now: the atomic's round trip overlaps the epilogue.
    int next = 0;
    if (lane == 0) next = atomicAdd(counter, 1);

    const float scale = gs[ge] * 1.1805916207174113e21f;  // 2^70: undoes the 2^-126 * 2^56 of the decode
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = nt * 8 + 2 * c + (i & 1);
        if (n >= cnt) continue;
        const long long row = row0 + n;
        const int r = g + 8 * (i >> 1);
        if (GATED) {
          const int col = 16 * wtile + r;
          const float gv = acc[0][nt][i] * scale, uv = acc[1][nt][i] * scale;
          if (split == 1) {
            const float h = gv / (1.f + __expf(-gv)) * uv;
            reinterpret_cast<u16*>(out)[row * N + col] = f2bf(h);
          } else {
            float* o = reinterpret_cast<float*>(out) + ((long long)sk * R + row) * (2 * N);
            o[col] = gv;
            o[N + col] = uv;
          }
        } else {
#pragma unroll
          for (int tt = 0; tt < 2; ++tt) {
            const int col = 32 * wtile + 16 * tt + r;
            const float y = acc[tt][nt][i] * scale * (1.f + __uint_as_float((u32)nw[col] << 16));
            if (split == 1)
              reinterpret_cast<u16*>(out)[row * N + col] = f2bf(y);
            else
              reinterpret_cast<float*>(out)[((long long)sk * R + row) * N + col] = y;
          }
        }
      }
    }
    item = warps + __shfl_sync(0xffffffffu, next, 0);
  }
}

// ---- route prep ----

// Route prep in one CTA: rank each local (token, slot) pair inside its expert with a shared
// atomic counter, prefix-sum the counts, then write the sorted token list, each pair's
// sorted position (-1 if not local), the blocks of <= BM pairs and [blocks, live rows].
// Pairs of one expert come out in arbitrary order; nothing downstream depends on it (rows
// are computed independently and _combine adds a token's slots in slot order).
#define ROUTE_THREADS 1024
#define ROUTE_PER_THREAD 4
extern "C" __global__ void __launch_bounds__(ROUTE_THREADS) moe_route(
    const int* __restrict__ ids, int R, int local_start, int n_local, int topk, int bm,
    int* __restrict__ tok, int* __restrict__ pos, int* __restrict__ blk_exp, int* __restrict__ blk_row,
    int* __restrict__ blk_cnt, int* __restrict__ nblk) {
  __shared__ int cnt[256], off[257], boff[257];
  GDC_LAUNCH();
  const int t = threadIdx.x;
  for (int e = t; e < n_local; e += ROUTE_THREADS) cnt[e] = 0;
  __syncthreads();
  int loc[ROUTE_PER_THREAD], rank[ROUTE_PER_THREAD];
#pragma unroll
  for (int i = 0; i < ROUTE_PER_THREAD; ++i) {
    const int r = t + i * ROUTE_THREADS;
    loc[i] = -1;
    if (r < R) {
      const int e = ids[r] - local_start;
      if (e >= 0 && e < n_local) {
        loc[i] = e;
        rank[i] = atomicAdd(&cnt[e], 1);
      }
    }
  }
  __syncthreads();
  if (t == 0) {
    int o = 0, b = 0;
    for (int e = 0; e < n_local; ++e) {
      off[e] = o;
      boff[e] = b;
      o += cnt[e];
      b += (cnt[e] + bm - 1) / bm;
    }
    off[n_local] = o;
    boff[n_local] = b;
    nblk[0] = b;
    nblk[1] = o;
    nblk[2] = 0;  // moe_gemm1's and moe_gemm2's item counters, zero for this call
    nblk[3] = 0;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < ROUTE_PER_THREAD; ++i) {
    const int r = t + i * ROUTE_THREADS;
    if (r < R) {
      int p = -1;
      if (loc[i] >= 0) {
        p = off[loc[i]] + rank[i];
        tok[p] = r / topk;
      }
      pos[r] = p;
    }
  }
  for (int e = t; e < n_local; e += ROUTE_THREADS) {
    for (int k = 0; k < boff[e + 1] - boff[e]; ++k) {
      const int b = boff[e] + k;
      blk_exp[b] = e;
      blk_row[b] = off[e] + k * bm;
      blk_cnt[b] = min(bm, cnt[e] - k * bm);
    }
  }
}

// ---- entry points: GATED = GEMM1 (W13 + SwiGLU), EXACT = exact scale decode ----

#define GEMM_ARGS const u16* x, const int* tok, const int* blk_exp, const int* blk_row, \
  const int* blk_cnt, const int* nblk, const uint4* w, const uint4* sf, const float* gs, \
  const u16* nw, void* out, int R, int local_start, int K, int N, int split
#define GEMM_PASS x, tok, blk_exp, blk_row, blk_cnt, nblk, w, sf, gs, nw, out, R, local_start, K, N, split

extern "C" __global__ void __launch_bounds__(128, 4) moe_gemm1(GEMM_ARGS) { gemm<true, false>(GEMM_PASS); }
extern "C" __global__ void __launch_bounds__(128, 4) moe_gemm1_exact(GEMM_ARGS) { gemm<true, true>(GEMM_PASS); }
extern "C" __global__ void __launch_bounds__(128, 4) moe_gemm2(GEMM_ARGS) { gemm<false, false>(GEMM_PASS); }
extern "C" __global__ void __launch_bounds__(128, 4) moe_gemm2_exact(GEMM_ARGS) { gemm<false, true>(GEMM_PASS); }
"""
CUDA_SRC = (CUDA_SRC.replace("@DECODE@", _decode_source())
            .replace("@SCALE_FAST_MASK@", f"0x{SCALE_FAST_MASK:08X}")
            .replace("@SCALE_FAST_BIAS@", f"0x{SCALE_FAST_BIAS:08X}"))


# ---- Weight packing (prepare) ------------------------------------------------------------------

def pack_codes(w: torch.Tensor) -> torch.Tensor:
    """uint8 [E, R, K/2] (two E2M1 codes per byte, low nibble first) -> int32
    [E, R/16, K/64, 32, 4]: for each 16-row tile and 64-k chunk, 32 lanes x 4 words. Word kk
    of a lane holds its 8 codes for k16 step kk: register j, half b (b = 0: the lower k of
    the pair) at sign bit CODE_SIGN_BIT[j] + 16 b and magnitude bits CODE_MAG_LOW[j] + 16 b
    and up. Every bit of every code is kept: a reorder of bits, not of values."""
    e, r, kh = w.shape
    k = 2 * kh
    codes = torch.stack((w & 0xF, w >> 4), dim=-1).reshape(e, r, k)          # [E, R, K]
    # rows: t, hr, g   k: chunk, c, kk, h, b
    c6 = codes.view(e, r // TILE, 2, 8, k // CHUNK, 4, 4, 2, 2)
    # -> [E, t, chunk, g, c, kk, b, h, hr]; register j = 2h + hr
    t = c6.permute(0, 1, 4, 3, 5, 6, 8, 7, 2)
    t = t.reshape(e, r // TILE, k // CHUNK, 32, 4, 2, 4).to(torch.int64)     # [..., kk, b, j]
    dev = w.device
    half = 16 * torch.arange(2, device=dev, dtype=torch.int64)[:, None]
    sign_at = torch.tensor(CODE_SIGN_BIT, device=dev, dtype=torch.int64)[None, :] + half
    mag_at = torch.tensor(CODE_MAG_LOW, device=dev, dtype=torch.int64)[None, :] + half
    words = (((t >> 3) & 1) << sign_at) | ((t & 7) << mag_at)
    words = words.sum(dim=(-1, -2))                                           # disjoint bits: sum = or
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words)
    return words.to(torch.int32).contiguous()


def recode_scales(sf: torch.Tensor) -> torch.Tensor:
    """Positive normal E4M3 bits b = 0 e3 e2 e1 e0 m2 m1 m0 -> X = e0 m2 m1 m0 0 e3 e2 e1,
    the fast path's 8-bit encoding (see scale_of). A bijection on these bytes: the same
    information, the same 8 bits."""
    b = sf.to(torch.int32)
    x = ((b & 0x08) << 4) | ((b & 0x07) << 4) | ((b >> 4) & 0x07)
    return x.to(torch.uint8)


def pack_scales(sf: torch.Tensor) -> torch.Tensor:
    """uint8 scale bytes [E, R, K/16] -> uint8 [E, R/16, K/512, 32, 8, 2]: per lane = 4g + c
    and per chunk of a group of 8, the pair (scale of row g, scale of row g + 8) of block
    4 * chunk + c."""
    e, r, kb = sf.shape
    s = sf.view(e, r // TILE, 2, 8, kb // (4 * GROUP), GROUP, 4)             # t, hr, g, grp, ch, c
    s = s.permute(0, 1, 4, 3, 6, 5, 2)                                        # t, grp, g, c, ch, hr
    return s.reshape(e, r // TILE, kb // (4 * GROUP), 32, GROUP, 2).contiguous()


# ---- CUDA compile and launch -------------------------------------------------------------------

CUDA_OPTIONS = ["-std=c++17"]


# Kernel arguments are only tensors and ints: torch.cuda._compile_kernel passes a Python
# float as a C float in torch 2.9 but as a C double in 2.14, so a float parameter would
# read garbage on one of them (it read 0.0 on the judge: every output was zero).
def _compile_cuda():
    compile_kernel = getattr(torch.cuda, "compile_kernel", None) or torch.cuda._compile_kernel
    names = ("moe_route", "moe_gemm1", "moe_gemm1_exact", "moe_gemm2", "moe_gemm2_exact")
    return {n: compile_kernel(kernel_source=CUDA_SRC, kernel_name=n, nvcc_options=CUDA_OPTIONS) for n in names}


def _pack_per_expert(w: torch.Tensor) -> torch.Tensor:
    """pack_codes one expert at a time: the int64 intermediate of a whole bank is ~70 GB."""
    out = torch.empty((w.shape[0], w.shape[1] // TILE, 2 * w.shape[2] // CHUNK, 32, 4), dtype=torch.int32,
                      device=w.device)
    for e in range(w.shape[0]):
        out[e] = pack_codes(w[e:e + 1])[0]
    return out


# Programmatic dependent launch for the CUDA kernels: torch's compiled kernel launches with
# cuLaunchKernel, which takes no attributes, so _launch_pdl calls cuLaunchKernelEx on the same
# CUfunction with CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION (6).
# On the H200 (check af6d1b9c, CUDA graph, cold L2) the chain saves 2-3 us at bs 1-8 and ~1 us at
# bs 16, and costs ~3 us at bs 32 (12 busy experts: the waits hold back the GEMMs' full grids).
PDL_MAX_BS = 16


# ctypes mirrors of CUlaunchAttribute and CUlaunchConfig (cuda.h).
class _LaunchAttrValue(ctypes.Union):
    _fields_ = [("pad", ctypes.c_char * 64), ("i", ctypes.c_int), ("align", ctypes.c_void_p)]


class _LaunchAttr(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("pad", ctypes.c_char * 4), ("value", _LaunchAttrValue)]


class _LaunchConfig(ctypes.Structure):
    _fields_ = [("grid_x", ctypes.c_uint), ("grid_y", ctypes.c_uint), ("grid_z", ctypes.c_uint),
                ("block_x", ctypes.c_uint), ("block_y", ctypes.c_uint), ("block_z", ctypes.c_uint),
                ("shared_mem", ctypes.c_uint), ("stream", ctypes.c_void_p),
                ("attrs", ctypes.POINTER(_LaunchAttr)), ("num_attrs", ctypes.c_uint)]


_PDL_ATTR = _LaunchAttr(id=6)
_PDL_ATTR.value.i = 1


_LIBCUDA = []  # torch's libcuda handle, loaded once


def _libcuda():
    if not _LIBCUDA:
        _LIBCUDA.append(torch.cuda._utils._get_cuda_library())
    return _LIBCUDA[0]


def _launch_pdl(kernel, grid, block, args) -> None:
    """kernel(grid=grid, block=block, args=args), launched as a programmatic dependent of the
    previous kernel in the stream. Tensors are passed as pointers and ints as C ints, as
    torch's launcher does."""
    keep = []
    for a in args:
        keep.append(ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else ctypes.c_int(a))
    params = (ctypes.c_void_p * len(keep))(*[ctypes.cast(ctypes.byref(k), ctypes.c_void_p) for k in keep])
    cfg = _LaunchConfig(grid[0], grid[1], grid[2], block[0], block[1], block[2], 0,
                        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), ctypes.pointer(_PDL_ATTR), 1)
    err = _libcuda().cuLaunchKernelEx(ctypes.byref(cfg), kernel.func, params, None)
    if err != 0:
        raise RuntimeError(f"cuLaunchKernelEx failed: CUresult {err}")


def _launch(kernel, grid, block, args, pdl: bool) -> None:
    if pdl:
        _launch_pdl(kernel, grid, block, args)
    else:
        kernel(grid=grid, block=block, args=args)


# ---- Triton kernels: route prep (few or many pairs), split-K SwiGLU reduce, norm + combine -----

POST_NORM_EPS = 1e-8
# Tokens per block of one expert (two n8 MMA tiles).
BM = 16
# Route prep by pair count: _route_prep (Triton, one pass of <= 64 pairs) is faster for
# few pairs (2-3 us against ~4 for moe_route on the H100 bench); moe_route (CUDA) from there up to
# 1024 threads x 4 pairs; beyond, _route_prep again (pairs per step PREP_CHUNK).
ROUTE_MIN_PAIRS = 64
ROUTE_MAX_PAIRS = 4096
PREP_CHUNK = 256


@triton.jit
def _route_prep(ids_ptr, R, local_start, tok_ptr, pos_ptr, blk_exp_ptr, blk_row_ptr, blk_cnt_ptr, nblk_ptr,
                N_LOCAL: tl.constexpr, BLOCK_E: tl.constexpr, CHUNK: tl.constexpr, BLOCK_B: tl.constexpr,
                BM: tl.constexpr, TOPK: tl.constexpr, GDC: tl.constexpr = False):
    """Sort the local pairs by expert (stable in pair order) and cut each expert into blocks
    of BM. Outputs: tok[i] = token of the i-th sorted pair; pos[r] = sorted position of pair
    r (= token * TOPK + slot), -1 if not local; per block its expert, first sorted row and
    row count; nblk = [blocks, live sorted rows, 0, 0] (the last two are the GEMMs' item
    counters)."""
    if GDC:
        gdc_launch_dependents()
    e = tl.arange(0, BLOCK_E)
    counts = tl.zeros([BLOCK_E], dtype=tl.int32)
    for c0 in range(0, R, CHUNK):
        r = c0 + tl.arange(0, CHUNK)
        loc = tl.load(ids_ptr + r, mask=r < R, other=-1) - local_start
        valid = (r < R) & (loc >= 0) & (loc < N_LOCAL)
        hit = (loc[None, :] == e[:, None]) & valid[None, :]
        counts += tl.sum(hit.to(tl.int32), axis=1)

    nb = (counts + BM - 1) // BM
    blk_end = tl.cumsum(nb, axis=0)
    blk_beg = blk_end - nb
    row_beg = tl.cumsum(counts, axis=0) - counts
    total = tl.sum(nb, axis=0)
    tl.store(nblk_ptr, total)
    tl.store(nblk_ptr + 1, tl.sum(counts, axis=0))  # live sorted rows
    tl.store(nblk_ptr + 2, 0)  # moe_gemm1's and moe_gemm2's item counters, zero for this call
    tl.store(nblk_ptr + 3, 0)

    # Block b belongs to the first expert whose block range ends after b.
    b = tl.arange(0, BLOCK_B)
    owner = tl.sum((blk_end[:, None] <= b[None, :]).to(tl.int32), axis=0)
    sel = e[:, None] == owner[None, :]
    first = tl.sum(tl.where(sel, blk_beg[:, None], 0), axis=0)
    rbeg = tl.sum(tl.where(sel, row_beg[:, None], 0), axis=0)
    cnt = tl.sum(tl.where(sel, counts[:, None], 0), axis=0)
    k = b - first
    in_use = b < total
    tl.store(blk_exp_ptr + b, owner, mask=in_use)
    tl.store(blk_row_ptr + b, rbeg + k * BM, mask=in_use)
    tl.store(blk_cnt_ptr + b, tl.minimum(cnt - k * BM, BM), mask=in_use)

    run = row_beg
    for c0 in range(0, R, CHUNK):
        r = c0 + tl.arange(0, CHUNK)
        loc = tl.load(ids_ptr + r, mask=r < R, other=-1) - local_start
        valid = (r < R) & (loc >= 0) & (loc < N_LOCAL)
        hit = (loc[None, :] == e[:, None]) & valid[None, :]
        hit_i = hit.to(tl.int32)
        rank = tl.cumsum(hit_i, axis=1) - 1 + run[:, None]
        pos = tl.sum(tl.where(hit, rank, 0), axis=0)
        tl.store(tok_ptr + pos, r // TOPK, mask=valid)
        tl.store(pos_ptr + r, tl.where(valid, pos, -1), mask=r < R)
        run += tl.sum(hit_i, axis=1)


@triton.jit
def _swiglu_reduce(part_ptr, inter_ptr, nblk_ptr, R, I: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr,
                   GDC: tl.constexpr = False):
    """Split-K only: inter[row] = silu(sum_s gate_s) * sum_s up_s, bf16, from
    moe_gemm1's fp32 slices [SPLIT, R, 2I]. Only the live sorted rows (nblk[1], from
    the route prep) are read and written; the rest of the grid does nothing."""
    if GDC:
        gdc_launch_dependents()
        gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    live = row < tl.load(nblk_ptr + 1)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.zeros([BLOCK], dtype=tl.float32)
    u = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        src = part_ptr + (s * R + row) * (2 * I) + c
        g += tl.load(src, mask=live, other=0.0)
        u += tl.load(src + I, mask=live, other=0.0)
    tl.store(inter_ptr + row * I + c, (g * tl.sigmoid(g) * u).to(tl.bfloat16), mask=live)


@triton.jit
def _combine(y_ptr, pos_ptr, wts_ptr, out_ptr, eps, R,
             H: tl.constexpr, TOPK: tl.constexpr, KP: tl.constexpr, Y_SPLIT: tl.constexpr, GDC: tl.constexpr = False):
    """out[t] = sum over t's local pairs of g * rmsnorm(y[pair]); zero if none is local.
    With Y_SPLIT > 1, y[pair] is the sum of moe_gemm2's fp32 slices. All of the token's
    rows are loaded as one [KP, H] tile (KP = TOPK rounded up to a power of 2), so the
    norms take one reduction instead of one per slot."""
    if GDC:
        gdc_wait()
    t = tl.program_id(0)
    h = tl.arange(0, H)
    k = tl.arange(0, KP)
    p = tl.load(pos_ptr + t * TOPK + k, mask=k < TOPK, other=-1)
    ok = p >= 0
    rows = tl.maximum(p, 0).to(tl.int64)[:, None]
    y = tl.zeros([KP, H], dtype=tl.float32)
    for z in tl.static_range(Y_SPLIT):
        y += tl.load(y_ptr + (z * R + rows) * H + h[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    g = tl.load(wts_ptr + t * TOPK + k, mask=ok, other=0.0)
    scale = tl.where(ok, g * tl.rsqrt(tl.sum(y * y, axis=1) / H + eps), 0.0)
    acc = tl.sum(y * scale[:, None], axis=0)
    tl.store(out_ptr + t.to(tl.int64) * H + h, acc.to(tl.bfloat16))


# ---- prepare ---------------------------------------------------------------------------------------

def _scale_bytes(sf: torch.Tensor) -> torch.Tensor:
    """The scale bytes a GEMM reads: E4M3 as is for the exact path, recoded for the fast one."""
    b = sf.view(torch.uint8)
    return b if _needs_exact(sf) else recode_scales(b)


def _needs_exact(sf: torch.Tensor) -> bool:
    """True if any E4M3 scale is zero, subnormal, negative or NaN: recode_scales and the
    fast scale_of<false> are right only for positive normal finite scales.
    The exact path decodes an E4M3 NaN (0x7F, 0xFF) as +-480, finite; nvfp4.py clamps every
    scale to at most 448, so the judge's weights have no NaN scale."""
    b = sf.view(torch.uint8)
    return bool((((b & 0x78) == 0) | ((b & 0x80) != 0) | ((b & 0x7F) == 0x7F)).any())


def _warm(state, hid, n_experts) -> None:
    """Compile the few-token path's Triton specializations (GDC=True) outside the timed calls:
    one call at bs 1, 2, 8, 16, routes on local experts."""
    dev = state["norm_w"].device
    for bs in (1, 2, 8, 16):
        ids = (torch.arange(bs * 8, device=dev, dtype=torch.int32).view(bs, 8) * 5) % n_experts
        data = {"hidden_states": torch.zeros((bs, hid), dtype=torch.bfloat16, device=dev), "topk_ids": ids,
                "topk_weights": torch.full((bs, 8), 0.125, device=dev),
                "config": {"bs": bs, "d_hidden": hid, "d_expert": state["inter"], "n_experts": n_experts,
                           "local_start": 0, "local_count": min(32, n_experts), "top_k": 8, "sharding": "ep8"}}
        kernel_fn(data, state)
    torch.cuda.synchronize()


def prepare(weights: Weights):
    """Compiles the CUDA GEMMs (NVRTC, in process) and repacks the NVFP4 codes and E4M3
    scales into their fragment layout (pack_codes, pack_scales: a reorder of bits, the
    values and their precision untouched). post_norm_w stays BF16; the kernels add the 1.
    Each GEMM takes the exact scale decode only if its scales need it (_needs_exact)."""
    dev = weights["w13"].device
    hid, inter = weights["w13"].shape[2] * 2, weights["w2"].shape[2] * 2
    assert hid % (CHUNK * GROUP) == 0 and inter % (CHUNK * GROUP) == 0, "K must be a multiple of 512"
    assert inter % 16 == 0 and hid % 32 == 0, "GEMM1 tiles 16 columns of I, GEMM2 32 rows of H"
    assert hid & (hid - 1) == 0, "_combine reads a whole row with tl.arange(0, H): H must be a power of 2"
    state = {
        "w13_exact": _needs_exact(weights["w13_sf"]),
        "w2_exact": _needs_exact(weights["w2_sf"]),
        "inter": inter,
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2_gs": weights["w2_gs"].float().contiguous(),
        "cuda": _compile_cuda(),
        "w13": _pack_per_expert(weights["w13"]),
        "w13_sf": pack_scales(_scale_bytes(weights["w13_sf"])),
        "w2": _pack_per_expert(weights["w2"]),
        "w2_sf": pack_scales(_scale_bytes(weights["w2_sf"])),
        "norm_w": weights["post_norm_w"].contiguous(),
        "num_sms": torch.cuda.get_device_properties(dev).multi_processor_count,
    }
    _warm(state, hid, weights["w13"].shape[0])
    return state


# ---- kernel_fn: the timed call ------------------------------------------------------------------

def _split_k(bs: int, k: int) -> int:
    """K slices per work item, by token-count range. With few tokens only ~2-3 experts are
    touched and there are too few items for the GPU's warps; splitting K multiplies the
    items and shortens each warp's chain. A slice is a whole number of 512-k groups, so the
    count must divide k / 512: otherwise some groups would be summed twice.
    For 7 <= bs <= 64 the items are near-uniform (a whole K slice of a weight tile), so the
    last round of the persistent grid is partly idle; 2 slices shrink that tail (+4.7% on
    H200; 4 slices here, or 2 slices also above bs 64, measured the same)."""
    want = 4 if bs <= 2 else 2 if bs <= 64 else 1
    groups = k // (CHUNK * GROUP)
    while groups % want:
        want -= 1
    return want


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    """Route prep, moe_gemm1, [_swiglu_reduce], moe_gemm2, _combine on the current stream.
    Every buffer is allocated per call (CUDA-graph safe); the only branches are on shapes,
    config and the exact-path flags prepare() set."""
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()  # no-op for the int32 the problem sends
    wts = data["topk_weights"].float().contiguous()
    m_tok, hid = x.shape
    top_k = ids.shape[1]
    inter_dim = state["inter"]
    n_local, start = cfg["local_count"], cfg["local_start"]
    r = m_tok * top_k
    max_blocks = min(r, triton.cdiv(r, BM) + n_local)
    dev = x.device

    pdl = m_tok <= PDL_MAX_BS
    tok = torch.empty(r, dtype=torch.int32, device=dev)
    pos = torch.empty(r, dtype=torch.int32, device=dev)
    blk = torch.empty((3, max_blocks), dtype=torch.int32, device=dev)
    # [blocks, live sorted rows, moe_gemm1 item counter, moe_gemm2 item counter]
    nblk = torch.empty(4, dtype=torch.int32, device=dev)
    if ROUTE_MIN_PAIRS < r <= ROUTE_MAX_PAIRS and n_local <= 256:
        state["cuda"]["moe_route"](grid=(1, 1, 1), block=(1024, 1, 1),
                                   args=[ids, r, start, n_local, top_k, BM, tok, pos, blk[0], blk[1], blk[2], nblk])
    else:
        _route_prep[(1,)](ids, r, start, tok, pos, blk[0], blk[1], blk[2], nblk,
                          N_LOCAL=n_local, BLOCK_E=triton.next_power_of_2(n_local),
                          CHUNK=min(PREP_CHUNK, triton.next_power_of_2(r)),
                          BLOCK_B=triton.next_power_of_2(max_blocks), BM=BM, TOPK=top_k, GDC=pdl,
                          num_warps=4 if r <= 128 else 8)

    s1 = _split_k(m_tok, hid)
    s2 = _split_k(m_tok, inter_dim)
    inter = torch.empty((r, inter_dim), dtype=torch.bfloat16, device=dev)
    h = inter if s1 == 1 else torch.empty((s1, r, 2 * inter_dim), dtype=torch.float32, device=dev)
    # One warp per item: (block, 16 inter columns, K slice); 4 warps per CTA.
    kern = state["cuda"]["moe_gemm1_exact" if state["w13_exact"] else "moe_gemm1"]
    cg = min(state["num_sms"] * CUDA_CTAS_PER_SM, triton.cdiv(max_blocks * (inter_dim // 16) * s1, 4))
    _launch(kern, (cg, 1, 1), (128, 1, 1),
            [x, tok, blk[0], blk[1], blk[2], nblk, state["w13"], state["w13_sf"], state["w13_gs"],
             state["norm_w"], h, r, start, hid, inter_dim, s1], pdl)
    if s1 > 1:
        blk_i = min(1024, inter_dim)
        _swiglu_reduce[(r, inter_dim // blk_i)](h, inter, nblk, r, I=inter_dim, SPLIT=s1, BLOCK=blk_i, GDC=pdl,
                                                num_warps=4, **({"launch_pdl": True} if pdl else {}))
    y = torch.empty((r, hid) if s2 == 1 else (s2, r, hid), dtype=torch.bfloat16 if s2 == 1 else torch.float32,
                    device=dev)
    # One warp per item: (block, 32 output rows, K slice).
    kern = state["cuda"]["moe_gemm2_exact" if state["w2_exact"] else "moe_gemm2"]
    cg = min(state["num_sms"] * CUDA_CTAS_PER_SM, triton.cdiv(max_blocks * (hid // 32) * s2, 4))
    _launch(kern, (cg, 1, 1), (128, 1, 1),
            [inter, tok, blk[0], blk[1], blk[2], nblk, state["w2"], state["w2_sf"], state["w2_gs"],
             state["norm_w"], y, r, start, inter_dim, hid, s2], pdl)
    out = torch.empty((m_tok, hid), dtype=torch.bfloat16, device=dev)
    _combine[(m_tok,)](y, pos, wts, out, POST_NORM_EPS, r, H=hid, TOPK=top_k, KP=triton.next_power_of_2(top_k),
                       Y_SPLIT=s2, GDC=pdl, num_warps=16 if m_tok <= 64 else 8,
                       **({"launch_pdl": True} if pdl else {}))
    return out
