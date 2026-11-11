# okc: problem=muse_moe_w4a16
"""Routed experts with NVFP4 weights: two CUDA GEMMs (NVFP4 dequantized in registers,
mma.sync) between three small Triton kernels.

1. _route_prep (Triton, one CTA): groups this GPU's (token, expert) pairs by expert and
   cuts each expert into blocks of <= 16 tokens. Replaces expert_map + moe_align_block_size.
2. moe_gemm1 (CUDA): per block and 16 columns of I, x . Wgate^T and x . Wup^T; the
   epilogue applies the expert's global scale and writes silu(gate) * up in BF16.
3. moe_gemm2 (CUDA): per block and 32 rows of H, inter . W2^T; the epilogue applies the
   global scale and (1 + post_norm_w) and writes y.
4. _combine (Triton, one CTA per token): RMSNorm of each of the token's y rows, weighted sum.

GEMMs. Each warp is independent and persistent: it takes a work item (block, columns, K
slice), streams two 16-row tiles of weights from global memory with one 16-byte load per
lane per 64-k chunk, decodes them in registers and runs mma.sync m16n8k16 with the
weights as A ("swap AB": the weight rows fill the MMA's M side, the few tokens its N side).
No shared memory, no barriers.

Dequantization is exact. An E2M1 code s|e1 e0|m written into a BF16 as sign bit 15 and
e1 e0 m at bits 8..6 reads as e2m1 * 2^-126 (a subnormal for e = 0, which is what E2M1
means by e = 0); one bf16x2 multiply by block_scale * 2^56 gives e2m1 * block_scale * 2^-70,
exact (2 + 4 significant bits fit BF16's 8), and the epilogue multiplies the FP32
accumulator by global_scale * 2^70. The E4M3 block scale becomes block_scale * 2^56 with
integer ops (0x5800 | (b & 0x7F) << 4) when every scale of the tensor is positive and
normal, and with two exact FP32 multiplies otherwise (prepare() checks).

Layout (pack_codes, pack_scales): prepare() reorders the bytes, never the values. Inside
each 64-k chunk the MMA fragment position (step kk, lane c, half h, b) reads the real k
16c + 4kk + 2h + b (the sum over k doesn't care about order), so a lane's weights for a
chunk are one 16-byte load, its 16 activations per token are contiguous, and its 16 k
fall in one E4M3 block: one scale per lane, row and chunk.

Split-K (bs <= 6): with 1-6 tokens only a few experts are touched and there are too few
work items to fill the GPU; each covers 1/2 or 1/4 of K and the slices are summed in FP32
(_swiglu_reduce for W13, _combine for W2).

The Triton GEMMs (_gemm1_swiglu, _gemm2) compute the same thing; they run only where the
CUDA ones can't (USE_CUDA = False), e.g. the CPU interpreter test.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl


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


print(f"muse_moe_w4a16: torch {torch.__version__}, triton {triton.__version__}")

# The CUDA GEMMs need a GPU with BF16 mma.sync; the CPU interpreter test sets False.
USE_CUDA = True
CUDA_CTAS_PER_SM = 3  # 128-thread CTAs; ~136 registers per thread
CHUNK, TILE, GROUP = 64, 16, 8

CUDA_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;

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

#ifndef OKC_EMULATE
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
#else
// Test-only build for GPUs without BF16 MMA (the dev box's sm75): the same arithmetic in
// FP32 and warp shuffles. Never used by the judge.
__device__ __forceinline__ float bf_lo(u32 v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(u32 v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ u16 f2bf(float v) {
  u32 u = __float_as_uint(v);
  return (u16)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ u32 hmul2(u32 a, u32 b) {
  return (u32)f2bf(bf_lo(a) * bf_lo(b)) | ((u32)f2bf(bf_hi(a) * bf_hi(b)) << 16);
}
__device__ __noinline__ void mma(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  for (int i = 0; i < 4; ++i) {
    const int row = g + 8 * (i >> 1), col = 2 * c + (i & 1);
    float sum = 0.f;
    for (int kq = 0; kq < 8; ++kq) {  // k = 2 kq + {0, 1}
      const int src_a = 4 * (row & 7) + (kq & 3), src_b = 4 * col + (kq & 3);
      const u32 x0 = __shfl_sync(0xffffffffu, a0, src_a), x1 = __shfl_sync(0xffffffffu, a1, src_a);
      const u32 x2 = __shfl_sync(0xffffffffu, a2, src_a), x3 = __shfl_sync(0xffffffffu, a3, src_a);
      const u32 y0 = __shfl_sync(0xffffffffu, b0, src_b), y1 = __shfl_sync(0xffffffffu, b1, src_b);
      const u32 av = kq < 4 ? (row < 8 ? x0 : x1) : (row < 8 ? x2 : x3);
      const u32 bv = kq < 4 ? y0 : y1;
      sum += bf_lo(av) * bf_lo(bv) + bf_hi(av) * bf_hi(bv);
    }
    d[i] += sum;
  }
}
#endif

// (scale, scale) as bf16x2, equal to E4M3 scale * 2^56. Fast path: normal positive E4M3,
// bits 0x5800 | (b & 0x7F) << 4. EXACT: also subnormal/negative, via two exact FP32 mults.
template <bool EXACT>
__device__ __forceinline__ u32 scale_pair(u32 b) {
  if (EXACT) {
    float f = __int_as_float(((b & 0x7Fu) << 20) | ((b & 0x80u) << 24));
    f = f * 1.329227995784916e36f;   // 2^120
    f = f * 7.205759403792794e16f;   // 2^56
    u32 h = __float_as_uint(f) >> 16;
    return h | (h << 16);
  } else {
    u32 v = __byte_perm(b, 0, 0x4040);
    return ((v & 0x007F007Fu) << 4) | 0x58005800u;
  }
}

// Decode word q (8 E2M1 codes, marlin_pack layout) into the four A registers and scale them.
__device__ __forceinline__ void decode(u32 q, u32 pg, u32 pg8, u32& r0, u32& r1, u32& r2, u32& r3) {
  r0 = hmul2((q & 0x80008000u) | ((q >> 6) & 0x01C001C0u), pg);
  r1 = hmul2(((q << 4) & 0x80008000u) | ((q >> 2) & 0x01C001C0u), pg8);
  r2 = hmul2(((q << 8) & 0x80008000u) | ((q << 2) & 0x01C001C0u), pg);
  r3 = hmul2(((q << 12) & 0x80008000u) | ((q << 6) & 0x01C001C0u), pg8);
}

__device__ __forceinline__ u32 word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// One work item: NT n8 token tiles, two A tiles (t0, t1), chunks [ch0, ch1).
template <int NT, bool EXACT>
__device__ __forceinline__ void tile_loop(
    const uint4* __restrict__ wt0, const uint4* __restrict__ wt1,
    const uint4* __restrict__ st0, const uint4* __restrict__ st1,
    const uint4* const* xrow, int lane, int ch0, int ch1, float acc[2][2][4]) {
  const int c = lane & 3;
  for (int grp = ch0 >> 3; grp < (ch1 >> 3); ++grp) {
    const uint4 s0 = ldg_keep(st0 + grp * 32);
    const uint4 s1 = ldg_keep(st1 + grp * 32);
    uint4 a0[8], a1[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      a0[u] = ldg_stream(wt0 + (grp * 8 + u) * 32);
      a1[u] = ldg_stream(wt1 + (grp * 8 + u) * 32);
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int ch = grp * 8 + u;
      uint4 b[NT][2];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        b[nt][0] = ldg_keep(xrow[nt] + ch * 8 + 2 * c);
        b[nt][1] = ldg_keep(xrow[nt] + ch * 8 + 2 * c + 1);
      }
      const u32 sw0 = (word(s0, u >> 1) >> ((u & 1) * 16)) & 0xFFFFu;
      const u32 sw1 = (word(s1, u >> 1) >> ((u & 1) * 16)) & 0xFFFFu;
      const u32 p0g = scale_pair<EXACT>(sw0 & 0xFFu), p0g8 = scale_pair<EXACT>(sw0 >> 8);
      const u32 p1g = scale_pair<EXACT>(sw1 & 0xFFu), p1g8 = scale_pair<EXACT>(sw1 >> 8);
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
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, c = lane & 3;
  const int warps = (gridDim.x * blockDim.x) >> 5;
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int KC = K >> 6;
  const int tiles = (GATED ? 2 * N : N) >> 4;        // 16-row A tiles per expert
  const int nwt = GATED ? (N >> 4) : (N >> 5);       // warp tiles
  const int nblk = *nblk_ptr;
  const int chs = KC / split;
  const long long items = (long long)nblk * nwt * split;
  for (long long item = gw; item < items; item += warps) {
    const int b = (int)(item % nblk);
    const int wtile = (int)((item / nblk) % nwt);
    const int sk = (int)(item / ((long long)nblk * nwt));
    const long long ge = blk_exp[b] + local_start;
    const int row0 = blk_row[b];
    const int cnt = blk_cnt[b];
    const int t0 = GATED ? wtile : 2 * wtile;
    const int t1 = GATED ? (N >> 4) + wtile : 2 * wtile + 1;
    const uint4* wt0 = w + ((ge * tiles + t0) * KC) * 32 + lane;
    const uint4* wt1 = w + ((ge * tiles + t1) * KC) * 32 + lane;
    const uint4* st0 = sf + ((ge * tiles + t0) * (KC >> 3)) * 32 + lane;
    const uint4* st1 = sf + ((ge * tiles + t1) * (KC >> 3)) * 32 + lane;
    const uint4* xrow[2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      int n = nt * 8 + g;
      n = n < cnt ? n : 0;  // padding tokens read a valid row; their columns are never stored
      const int row = GATED ? tok[row0 + n] : row0 + n;
      xrow[nt] = reinterpret_cast<const uint4*>(x + (long long)row * K);
    }
    float acc[2][2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
    if (cnt > 8)
      tile_loop<2, EXACT>(wt0, wt1, st0, st1, xrow, lane, sk * chs, (sk + 1) * chs, acc);
    else
      tile_loop<1, EXACT>(wt0, wt1, st0, st1, xrow, lane, sk * chs, (sk + 1) * chs, acc);

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
  }
}

#define GEMM_ARGS const u16* x, const int* tok, const int* blk_exp, const int* blk_row, \
  const int* blk_cnt, const int* nblk, const uint4* w, const uint4* sf, const float* gs, \
  const u16* nw, void* out, int R, int local_start, int K, int N, int split
#define GEMM_PASS x, tok, blk_exp, blk_row, blk_cnt, nblk, w, sf, gs, nw, out, R, local_start, K, N, split

extern "C" __global__ void __launch_bounds__(128) moe_gemm1(GEMM_ARGS) { gemm<true, false>(GEMM_PASS); }
extern "C" __global__ void __launch_bounds__(128) moe_gemm1_exact(GEMM_ARGS) { gemm<true, true>(GEMM_PASS); }
extern "C" __global__ void __launch_bounds__(128) moe_gemm2(GEMM_ARGS) { gemm<false, false>(GEMM_PASS); }
extern "C" __global__ void __launch_bounds__(128) moe_gemm2_exact(GEMM_ARGS) { gemm<false, true>(GEMM_PASS); }
"""


def pack_codes(w: torch.Tensor) -> torch.Tensor:
    """uint8 [E, R, K/2] (two E2M1 codes per byte, low nibble first) -> int32
    [E, R/16, K/64, 32, 4]: for each 16-row tile and 64-k chunk, 32 lanes x 4 words."""
    e, r, kh = w.shape
    k = 2 * kh
    codes = torch.stack((w & 0xF, w >> 4), dim=-1).reshape(e, r, k)          # [E, R, K]
    # rows: t, hr, g   k: chunk, c, kk, h, b
    c6 = codes.view(e, r // TILE, 2, 8, k // CHUNK, 4, 4, 2, 2)
    # -> [E, t, chunk, g, c, kk, b, j] with j = hr + 2h, then nibble n = 4b + 3 - j
    t = c6.permute(0, 1, 4, 3, 5, 6, 8, 7, 2)                                 # [.., b, h, hr]
    t = t.reshape(e, r // TILE, k // CHUNK, 8, 4, 4, 2, 4)                    # j = 2h + hr
    t = t.flip(-1)                                                            # index 3 - j
    t = t.reshape(e, r // TILE, k // CHUNK, 32, 4, 8).to(torch.int64)         # n = 4b + (3 - j)
    shifts = 4 * torch.arange(8, device=w.device, dtype=torch.int64)
    words = (t << shifts).sum(-1)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words)
    return words.to(torch.int32).contiguous()


def pack_scales(sf: torch.Tensor) -> torch.Tensor:
    """uint8 E4M3 bits [E, R, K/16] -> uint8 [E, R/16, K/512, 32, 8, 2]: per lane = 4g + c
    and per chunk of a group of 8, the pair (scale of row g, scale of row g + 8) of block
    4 * chunk + c."""
    e, r, kb = sf.shape
    s = sf.view(e, r // TILE, 2, 8, kb // (4 * GROUP), GROUP, 4)             # t, hr, g, grp, ch, c
    s = s.permute(0, 1, 4, 3, 6, 5, 2)                                        # t, grp, g, c, ch, hr
    return s.reshape(e, r // TILE, kb // (4 * GROUP), 32, GROUP, 2).contiguous()


# test_cuda_local.py adds -DOKC_EMULATE to run the kernels on a GPU without BF16 MMA.
CUDA_OPTIONS = ["-std=c++17"]


# Kernel arguments are only tensors and ints: torch.cuda._compile_kernel passes a Python
# float as a C float in torch 2.9 but as a C double in 2.14, so a float parameter would
# read garbage on one of them (it read 0.0 on the judge: every output was zero).
def _compile_cuda():
    names = ("moe_gemm1", "moe_gemm1_exact", "moe_gemm2", "moe_gemm2_exact")
    return {n: torch.cuda._compile_kernel(kernel_source=CUDA_SRC, kernel_name=n, nvcc_options=CUDA_OPTIONS)
            for n in names}


def _pack_per_expert(w: torch.Tensor) -> torch.Tensor:
    """pack_codes one expert at a time: the int64 intermediate of a whole bank is ~70 GB."""
    out = torch.empty((w.shape[0], w.shape[1] // TILE, 2 * w.shape[2] // CHUNK, 32, 4), dtype=torch.int32,
                      device=w.device)
    for e in range(w.shape[0]):
        out[e] = pack_codes(w[e:e + 1])[0]
    return out


POST_NORM_EPS = 1e-8
# Tile shape of both GEMMs: BN weight rows, BM tokens, BK reduction per step.
BN, BM, BK = 64, 16, 128
GEMM_WARPS, GEMM_STAGES = 4, 3
# Persistent CTAs per SM.
CTAS_PER_SM = 4
# Pairs per step of _route_prep's single CTA, and its warps, by pair count.
PREP_CHUNK = 256


# Dequantization trick. An E2M1 code s|e1 e0|m written into a BF16 as sign bit 15 and
# e1 e0 m at bits 8..6 is that BF16's exponent field e and top mantissa bit m: the value is
# exactly e2m1 * 2^-126 (e = 0 gives a subnormal, which is what E2M1 means by e = 0). One
# BF16 multiply by block_scale * 2^56 then gives e2m1 * block_scale * 2^-70, exact and
# normal; the epilogue multiplies the FP32 accumulator by global_scale * 2^70.
SCALE_SHIFT = tl.constexpr(2.0 ** 56)
ACC_SHIFT = tl.constexpr(2.0 ** (126 - 56))
# Four packed bytes per instance. $4 = bytes b0..b3; $5..$8 = (scale, scale) bf16x2 pairs
# for each byte. Output $k = bf16x2(low nibble of byte k, high nibble of byte k) * pair k:
# the two values at consecutive k. u = bytes >> 4 brings each high nibble to the low
# nibble of its byte; prmt puts byte k of the input in byte 0 and byte k of u in byte 2,
# and the shifts and masks place each nibble's sign at bit 15 and e1 e0 m at bits 8..6 of
# its half. herramientas/emula_asm.py checks the bit logic against the E2M1 grid.
_DEQUANT_PTX = tl.constexpr(
    "{\n.reg .b32 u, t, a, b;\nshr.b32 u, $4, 4;\n" + "".join(
        f"prmt.b32 t, $4, u, 0x0{4 + k}0{k};\nshl.b32 a, t, 6;\nand.b32 a, a, 0x01C001C0;\n"
        f"shl.b32 b, t, 12;\nand.b32 b, b, 0x80008000;\nor.b32 a, a, b;\nmul.rn.bf16x2 ${k}, a, ${5 + k};\n"
        for k in range(4)) + "}")
# 2^120: E4M3 bits placed at FP32 bits 26..20 read as exactly scale * 2^-120.
_E4M3_UNSHIFT = tl.constexpr(2.0 ** 120)
# The CPU interpreter can't run PTX; test_interprete.py sets this to False to run the same
# arithmetic in Triton ops.
DEQUANT_ASM = True


@triton.jit
def _scale_bf16(b, EXACT: tl.constexpr):
    """uint8 E4M3 block scales -> bf16 scale * 2^56, exact.

    Fast path (normal E4M3, which is every scale a quantizer emits unless a block is ~1e-5
    of the tensor's max): the BF16 exponent field must be e + 176 = 0b1011_eeee, so
    the BF16 bits are 0x5800 | e << 7 | m << 4 = 0x5800 | (b & 0x7F) << 4, plus the sign.
    Three instructions, no multiply.

    EXACT path, also right for subnormal E4M3: the same bits at FP32 bits 26..20 read as
    scale * 2^-120 (a subnormal FP32 when the E4M3 is), and two exact multiplies (PTX
    mul.f32 keeps subnormals) give scale * 2^56, whose top 16 bits are the BF16.
    prepare() picks EXACT per weight tensor."""
    b = b.to(tl.int32)
    if EXACT:
        f = (((b & 0x7F) << 20) | ((b & 0x80) << 24)).to(tl.float32, bitcast=True)
        f = f * _E4M3_UNSHIFT * SCALE_SHIFT
        bits = f.to(tl.int32, bitcast=True) >> 16
    else:
        bits = ((b & 0x7F) << 4) | 0x5800 | ((b & 0x80) << 8)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _dequant(p, s, BNT: tl.constexpr, BK: tl.constexpr, ASM: tl.constexpr, EXACT: tl.constexpr):
    """Packed tile p uint8 [BNT, BK // 2] in prepare()'s fragment order (logical byte
    16c + 2kb + h holds the codes of k = 16kb + 8h + 2c + {0, 1}) and its E4M3 block scales
    s uint8 [BNT, BK // 16] -> bf16 [BNT, BK] in natural k order, equal to
    e2m1 * block_scale * 2^-70 (the global scale is applied to the accumulator)."""
    KB: tl.constexpr = BK // 16
    BKH: tl.constexpr = BK // 2
    sb = _scale_bf16(s, EXACT)
    if ASM:
        bits = sb.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        pair = tl.reshape(tl.broadcast_to((bits | (bits << 16))[:, None, :, None], (BNT, 4, KB, 2)), (BNT, BKH))
        r = tl.inline_asm_elementwise(_DEQUANT_PTX, "=r,=r,=r,=r,r,r,r,r,r", [p, pair],
                                      dtype=tl.int32, is_pure=True, pack=4)
        lo = (r & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True)
        hi = (r >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)
    else:
        sf = tl.reshape(tl.broadcast_to(sb[:, None, :, None], (BNT, 4, KB, 2)), (BNT, BKH)).to(tl.float32)
        c = p.to(tl.int32)
        lo_bits = (((c & 7) << 6) | ((c & 8) << 12)) << 16
        hi_bits = ((((c >> 4) & 7) << 6) | (((c >> 4) & 8) << 12)) << 16
        lo = (lo_bits.to(tl.float32, bitcast=True) * sf).to(tl.bfloat16)
        hi = (hi_bits.to(tl.float32, bitcast=True) * sf).to(tl.bfloat16)
    d = tl.reshape(tl.join(lo, hi), (BNT, 4, KB, 2, 2))            # (c, kb, h, nibble)
    return tl.reshape(tl.permute(d, (0, 2, 3, 1, 4)), (BNT, BK))  # (kb, h, c, nibble) = k


@triton.jit
def _route_prep(ids_ptr, R, local_start, tok_ptr, pos_ptr, blk_exp_ptr, blk_row_ptr, blk_cnt_ptr, nblk_ptr,
                N_LOCAL: tl.constexpr, BLOCK_E: tl.constexpr, CHUNK: tl.constexpr, BLOCK_B: tl.constexpr,
                BM: tl.constexpr, TOPK: tl.constexpr):
    """Sort the local pairs by expert (stable in pair order) and cut each expert into blocks
    of BM. Outputs: tok[i] = token of the i-th sorted pair; pos[r] = sorted position of pair
    r (= token * TOPK + slot), -1 if not local; per block its expert, first sorted row and
    row count; nblk = number of blocks."""
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
def _gemm1_swiglu(x_ptr, tok_ptr, blk_exp_ptr, blk_row_ptr, blk_cnt_ptr, nblk_ptr,
                  w_ptr, sf_ptr, gs_ptr, out_ptr, local_start, num_progs, R,
                  K: tl.constexpr, I: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                  SPLIT: tl.constexpr, ASM: tl.constexpr, EXACT: tl.constexpr):
    """SPLIT == 1: out[row] = silu(x[tok] . Wg^T) * (x[tok] . Wu^T), bf16 [R, I].
    SPLIT > 1 (split-K, for few tokens): out[s, row] = gate and up sums over K slice s, fp32
    [SPLIT, R, 2I] (gate in [0, I), up in [I, 2I)); _swiglu_reduce finishes them."""
    BKH: tl.constexpr = BK // 2
    SB: tl.constexpr = BK // 16
    NT: tl.constexpr = I // BN
    KS: tl.constexpr = K // BK
    KSS: tl.constexpr = KS // SPLIT
    WT: tl.constexpr = 2 * BN * BKH  # bytes of one K step of a tile: gate rows, then up rows
    ST: tl.constexpr = 2 * BN * SB
    nblk = tl.load(nblk_ptr)
    n = tl.arange(0, BN)
    m = tl.arange(0, BM)
    j = tl.arange(0, BKH)
    q = tl.arange(0, BK)
    jb = tl.arange(0, BK // 16)
    for item in range(tl.program_id(0), nblk * NT * SPLIT, num_progs):
        # Consecutive items share the column tile, so blocks of one expert read it together.
        b = item % nblk
        nt = (item // nblk) % NT
        sk = item // (nblk * NT)
        col = nt * BN + n
        ge = (tl.load(blk_exp_ptr + b) + local_start).to(tl.int64)
        row0 = tl.load(blk_row_ptr + b)
        live = m < tl.load(blk_cnt_ptr + b)
        tok = tl.load(tok_ptr + row0 + m, mask=live, other=0)
        xs = x_ptr + tok[None, :].to(tl.int64) * K + q[:, None]  # [BK, BM], contiguous in k
        # Weights repacked by prepare(): [expert, N tile, K step, gate/up, BN, BKH].
        wg = w_ptr + ge * (2 * I * (K // 2)) + nt * (KS * WT) + n[:, None] * BKH + j[None, :]
        wu = wg + BN * BKH
        sg = sf_ptr + ge * (2 * I * (K // 16)) + nt * (KS * ST) + n[:, None] * SB + jb[None, :]
        su = sg + BN * SB
        acc_g = tl.zeros([BN, BM], dtype=tl.float32)
        acc_u = tl.zeros([BN, BM], dtype=tl.float32)
        for ks in range(sk * KSS, (sk + 1) * KSS):
            xt = tl.load(xs + ks * BK, mask=live[None, :], other=0.0)
            acc_g = tl.dot(_dequant(tl.load(wg + ks * WT), tl.load(sg + ks * ST), BN, BK, ASM, EXACT), xt, acc_g)
            acc_u = tl.dot(_dequant(tl.load(wu + ks * WT), tl.load(su + ks * ST), BN, BK, ASM, EXACT), xt, acc_u)
        gs = tl.load(gs_ptr + ge) * ACC_SHIFT
        rows = (row0 + m)[None, :].to(tl.int64)
        if SPLIT == 1:
            g = acc_g * gs
            h = g * tl.sigmoid(g) * (acc_u * gs)
            tl.store(out_ptr + rows * I + col[:, None], h.to(tl.bfloat16), mask=live[None, :])
        else:
            dst = out_ptr + (sk * R + rows) * (2 * I) + col[:, None]
            tl.store(dst, acc_g * gs, mask=live[None, :])
            tl.store(dst + I, acc_u * gs, mask=live[None, :])


@triton.jit
def _swiglu_reduce(part_ptr, inter_ptr, nblk_ptr, R, I: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    """Split-K only: inter[row] = silu(sum_s gate_s) * sum_s up_s, bf16, from
    _gemm1_swiglu's fp32 slices [SPLIT, R, 2I]. Only the live sorted rows (nblk[1], from
    _route_prep) are read and written; the rest of the grid does nothing."""
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
def _gemm2(a_ptr, blk_exp_ptr, blk_row_ptr, blk_cnt_ptr, nblk_ptr,
           w_ptr, sf_ptr, gs_ptr, nw_ptr, y_ptr, local_start, num_progs, R,
           K: tl.constexpr, H: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
           SPLIT: tl.constexpr, ASM: tl.constexpr, EXACT: tl.constexpr):
    """y[row] = (inter[row] . W2^T) * gs * (1 + post_norm_w) for every sorted row: bf16 [R, H]
    when SPLIT == 1, else fp32 partial sums per K slice, [SPLIT, R, H], added by _combine."""
    BKH: tl.constexpr = BK // 2
    SB: tl.constexpr = BK // 16
    NT: tl.constexpr = H // BN
    KS: tl.constexpr = K // BK
    KSS: tl.constexpr = KS // SPLIT
    nblk = tl.load(nblk_ptr)
    n = tl.arange(0, BN)
    m = tl.arange(0, BM)
    j = tl.arange(0, BKH)
    q = tl.arange(0, BK)
    jb = tl.arange(0, BK // 16)
    for item in range(tl.program_id(0), nblk * NT * SPLIT, num_progs):
        b = item % nblk
        nt = (item // nblk) % NT
        sk = item // (nblk * NT)
        col = nt * BN + n
        ge = (tl.load(blk_exp_ptr + b) + local_start).to(tl.int64)
        row0 = tl.load(blk_row_ptr + b)
        live = m < tl.load(blk_cnt_ptr + b)
        rows = (row0 + m)[None, :].to(tl.int64)
        # Weights repacked by prepare(): [expert, N tile, K step, BN, BKH].
        ws = w_ptr + ge * (H * (K // 2)) + nt * (KS * BN * BKH) + n[:, None] * BKH + j[None, :]
        ss = sf_ptr + ge * (H * (K // 16)) + nt * (KS * BN * SB) + n[:, None] * SB + jb[None, :]
        acc = tl.zeros([BN, BM], dtype=tl.float32)
        for ks in range(sk * KSS, (sk + 1) * KSS):
            xt = tl.load(a_ptr + rows * K + ks * BK + q[:, None], mask=live[None, :], other=0.0)
            w = _dequant(tl.load(ws + ks * (BN * BKH)), tl.load(ss + ks * (BN * SB)), BN, BK, ASM, EXACT)
            acc = tl.dot(w, xt, acc)
        y = acc * (tl.load(gs_ptr + ge) * ACC_SHIFT) * (1.0 + tl.load(nw_ptr + col).to(tl.float32))[:, None]
        if SPLIT == 1:
            tl.store(y_ptr + rows * H + col[:, None], y.to(tl.bfloat16), mask=live[None, :])
        else:
            tl.store(y_ptr + (sk * R + rows) * H + col[:, None], y, mask=live[None, :])


@triton.jit
def _combine(y_ptr, pos_ptr, wts_ptr, out_ptr, eps, R,
             H: tl.constexpr, TOPK: tl.constexpr, Y_SPLIT: tl.constexpr):
    """out[t] = sum over t's local pairs of g * rmsnorm(y[pair]); zero if none is local.
    With Y_SPLIT > 1, y[pair] is the sum of _gemm2's fp32 slices."""
    t = tl.program_id(0)
    h = tl.arange(0, H)
    acc = tl.zeros([H], dtype=tl.float32)
    for s in tl.static_range(TOPK):
        p = tl.load(pos_ptr + t * TOPK + s)
        ok = p >= 0
        row = tl.maximum(p, 0).to(tl.int64)
        y = tl.zeros([H], dtype=tl.float32)
        for z in tl.static_range(Y_SPLIT):
            y += tl.load(y_ptr + (z * R + row) * H + h, mask=ok & (h < H), other=0.0).to(tl.float32)
        scale = tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
        acc += tl.where(ok, tl.load(wts_ptr + t * TOPK + s) * scale, 0.0) * y
    tl.store(out_ptr + t.to(tl.int64) * H + h, acc.to(tl.bfloat16))


def _needs_exact(sf: torch.Tensor) -> bool:
    """True if any E4M3 scale is zero, subnormal, negative or NaN: the fast decode
    (0x5800 | (b & 0x7F) << 4) is right only for positive normal finite scales."""
    b = sf.view(torch.uint8)
    return bool((((b & 0x78) == 0) | ((b & 0x80) != 0) | ((b & 0x7F) == 0x7F)).any())


def _by_tiles(t: torch.Tensor, halves: int, row_chunk: int) -> torch.Tensor:
    """[E, halves * rows, cols] -> the same bytes ordered [E, row tile, col step, half,
    BN, row_chunk], so one K step of one tile (gate and up together for W13) is a single
    contiguous block. A lossless reorder; the values and their precision are untouched."""
    e, r, c = t.shape
    nt, ks = r // halves // BN, c // row_chunk
    t = t.view(e, halves, nt, BN, ks, row_chunk).permute(0, 2, 4, 1, 3, 5)
    return t.contiguous().view(e, r, c)


def _fragment_order(codes: torch.Tensor) -> torch.Tensor:
    """Inside each BK // 2-byte row chunk, byte 16c + 2kb + h <- byte 8kb + 4h + c
    (c < 4, h < 2, kb < BK // 16): the bytes a thread needs for its wgmma A fragments
    become contiguous. _dequant inverts it in registers."""
    e, r, c = codes.shape
    kb = BK // 16
    t = codes.view(e, r, c // (BK // 2), kb, 2, 4).permute(0, 1, 2, 5, 3, 4)
    return t.contiguous().view(e, r, c)


def prepare(weights: Weights):
    """Repacks the NVFP4 codes and E4M3 scales by tile (_by_tiles); scales are read as raw
    uint8 and decoded in the kernels. The norm weight is kept as FP32 (1 + w). Each GEMM
    takes the exact scale decode only if its scales contain a subnormal E4M3."""
    dev = weights["w13"].device
    return {
        "w13_exact": _needs_exact(weights["w13_sf"]),
        "w2_exact": _needs_exact(weights["w2_sf"]),
        "inter": weights["w2"].shape[2] * 2,
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2_gs": weights["w2_gs"].float().contiguous(),
        **({"cuda": _compile_cuda(),
            "w13": _pack_per_expert(weights["w13"]),
            "w13_sf": pack_scales(weights["w13_sf"].view(torch.uint8)),
            "w2": _pack_per_expert(weights["w2"]),
            "w2_sf": pack_scales(weights["w2_sf"].view(torch.uint8))} if USE_CUDA else
           {"w13": _by_tiles(_fragment_order(weights["w13"]), 2, BK // 2),
            "w13_sf": _by_tiles(weights["w13_sf"].view(torch.uint8), 2, BK // 16),
            "w2": _by_tiles(_fragment_order(weights["w2"]), 1, BK // 2),
            "w2_sf": _by_tiles(weights["w2_sf"].view(torch.uint8), 1, BK // 16)}),
        "norm_w": weights["post_norm_w"].contiguous(),  # BF16 as given; the kernels add 1
        "num_sms": torch.cuda.get_device_properties(dev).multi_processor_count if dev.type == "cuda" else 4,
    }


def _split_k(bs: int) -> int:
    """K slices per GEMM tile, by token-count range. With few tokens only ~2-3 experts are
    touched, a few hundred tiles for ~4 x 132 CTA slots, and each CTA's 32-step K chain is
    latency-bound; splitting K multiplies the tiles and shortens the chain."""
    return 4 if bs <= 2 else 2 if bs <= 6 else 1


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].contiguous()
    wts = data["topk_weights"].float().contiguous()
    m_tok, hid = x.shape
    top_k = ids.shape[1]
    inter_dim = state["inter"]
    n_local, start = cfg["local_count"], cfg["local_start"]
    r = m_tok * top_k
    max_blocks = min(r, triton.cdiv(r, BM) + n_local)
    dev = x.device

    tok = torch.empty(r, dtype=torch.int32, device=dev)
    pos = torch.empty(r, dtype=torch.int32, device=dev)
    blk = torch.empty((3, max_blocks), dtype=torch.int32, device=dev)
    nblk = torch.empty(2, dtype=torch.int32, device=dev)  # [blocks, live sorted rows]
    _route_prep[(1,)](ids, r, start, tok, pos, blk[0], blk[1], blk[2], nblk,
                      N_LOCAL=n_local, BLOCK_E=triton.next_power_of_2(n_local),
                      CHUNK=min(PREP_CHUNK, triton.next_power_of_2(r)),
                      BLOCK_B=triton.next_power_of_2(max_blocks), BM=BM, TOPK=top_k,
                      num_warps=4 if r <= 128 else 8)

    progs = state["num_sms"] * CTAS_PER_SM
    # CUDA K slices are whole groups of 8 chunks of 64 (512 k); Triton's are BK steps.
    k_step = CHUNK * GROUP if USE_CUDA else BK
    s1 = min(_split_k(m_tok), hid // k_step)
    s2 = min(_split_k(m_tok), inter_dim // k_step)
    inter = torch.empty((r, inter_dim), dtype=torch.bfloat16, device=dev)
    h = inter if s1 == 1 else torch.empty((s1, r, 2 * inter_dim), dtype=torch.float32, device=dev)
    if USE_CUDA:
        # One warp per item: (block, 16 inter columns, K slice); 4 warps per CTA.
        kern = state["cuda"]["moe_gemm1_exact" if state["w13_exact"] else "moe_gemm1"]
        cg = min(state["num_sms"] * CUDA_CTAS_PER_SM, triton.cdiv(max_blocks * (inter_dim // 16) * s1, 4))
        kern(grid=(cg, 1, 1), block=(128, 1, 1),
             args=[x, tok, blk[0], blk[1], blk[2], nblk, state["w13"], state["w13_sf"], state["w13_gs"],
                   state["norm_w"], h, r, start, hid, inter_dim, s1])
    else:
        g1 = min(progs, max_blocks * (inter_dim // BN) * s1)
        _gemm1_swiglu[(g1,)](x, tok, blk[0], blk[1], blk[2], nblk,
                             state["w13"], state["w13_sf"], state["w13_gs"], h, start, g1, r,
                             K=hid, I=inter_dim, BN=BN, BM=BM, BK=BK, SPLIT=s1, ASM=DEQUANT_ASM,
                             EXACT=state["w13_exact"], num_warps=GEMM_WARPS, num_stages=GEMM_STAGES)
    if s1 > 1:
        blk_i = min(1024, inter_dim)
        _swiglu_reduce[(r, inter_dim // blk_i)](h, inter, nblk, r, I=inter_dim, SPLIT=s1, BLOCK=blk_i, num_warps=4)
    y = torch.empty((r, hid) if s2 == 1 else (s2, r, hid), dtype=torch.bfloat16 if s2 == 1 else torch.float32,
                    device=dev)
    if USE_CUDA:
        # One warp per item: (block, 32 output rows, K slice).
        kern = state["cuda"]["moe_gemm2_exact" if state["w2_exact"] else "moe_gemm2"]
        cg = min(state["num_sms"] * CUDA_CTAS_PER_SM, triton.cdiv(max_blocks * (hid // 32) * s2, 4))
        kern(grid=(cg, 1, 1), block=(128, 1, 1),
             args=[inter, tok, blk[0], blk[1], blk[2], nblk, state["w2"], state["w2_sf"], state["w2_gs"],
                   state["norm_w"], y, r, start, inter_dim, hid, s2])
    else:
        g2 = min(progs, max_blocks * (hid // BN) * s2)
        _gemm2[(g2,)](inter, blk[0], blk[1], blk[2], nblk,
                      state["w2"], state["w2_sf"], state["w2_gs"], state["norm_w"], y, start, g2, r,
                      K=inter_dim, H=hid, BN=BN, BM=BM, BK=BK, SPLIT=s2, ASM=DEQUANT_ASM,
                      EXACT=state["w2_exact"], num_warps=GEMM_WARPS, num_stages=GEMM_STAGES)
    out = torch.empty((m_tok, hid), dtype=torch.bfloat16, device=dev)
    _combine[(m_tok,)](y, pos, wts, out, POST_NORM_EPS, r, H=hid, TOPK=top_k, Y_SPLIT=s2, num_warps=8)
    return out
