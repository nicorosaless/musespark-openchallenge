# okc: problem=muse_moe_w4a16
"""Routed experts with NVFP4 weights: the route prep and two GEMMs in CUDA (NVFP4
dequantized in registers, mma.sync), the SwiGLU reduce of split-K and the norm in Triton.

1. moe_route (CUDA, one CTA) for 64 < pairs <= 4096, _route_prep (Triton) otherwise:
   groups this GPU's (token, expert) pairs by expert and cuts each expert into blocks of
   <= 16 tokens. Replaces expert_map + moe_align_block_size.
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

Split-K (bs <= 64): each work item covers 1/4 (bs <= 2) or 1/2 (bs <= 64) of K and the
slices are summed in FP32 (_swiglu_reduce for W13, _combine for W2). With few tokens there
are too few items to fill the GPU; with more, finer items shorten the idle tail of the
last round of the persistent grid.
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

# 128-thread CTAs, 4 per SM (16 warps): on H200 that beats 3 by ~3.5% (more warps for the
# near-uniform work items). The GEMMs' __launch_bounds__(128, 4) holds ptxas to 128
# registers so the 4 CTAs fit; the fast-path kernels don't spill there (the _exact ones,
# for subnormal/zero/negative scales, spill ~100 B: herramientas/compilar_cuda_sm90.py).
CUDA_CTAS_PER_SM = 4
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

// Decode word q (8 E2M1 codes, pack_codes layout) into the four A registers and scale them.
// Each register needs its codes' sign at bits 15/31 and exponent/mantissa at 8..6/24..22.
// Left shifts move low-half bits only into high-half positions outside the masks; the one
// right shift (by 6) moves high-half bits into low-half bits 10..15, outside its mask.
__device__ __forceinline__ void decode(u32 q, u32 pg, u32 pg8, u32& r0, u32& r1, u32& r2, u32& r3) {
  const u32 t = q << 3;
  r0 = hmul2(q & 0x81C081C0u, pg);                                        // sign 15, mag 8..6
  r1 = hmul2((t << 3) & 0x81C081C0u, pg8);                                // sign 9, mag 2..0
  r2 = hmul2(((t << 2) & 0x80008000u) | (t & 0x01C001C0u), pg);           // sign 10, mag 5..3
  r3 = hmul2(((q << 4) & 0x80008000u) | ((q >> 6) & 0x01C001C0u), pg8);   // sign 11, mag 14..12
}

__device__ __forceinline__ u32 word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// One work item: NT n8 token tiles, two A tiles (t0, t1), chunks [ch0, ch1).
template <int NT, bool EXACT>
__device__ __forceinline__ void tile_loop(
    const uint4* __restrict__ wt0, const uint4* __restrict__ wt1,
    const uint4* __restrict__ st0, const uint4* __restrict__ st1,
    const uint4* x0, const uint4* x1, int lane, int ch0, int ch1, float acc[2][2][4]) {
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
      b[0][0] = ldg_keep(x0 + ch * 8 + 2 * c);
      b[0][1] = ldg_keep(x0 + ch * 8 + 2 * c + 1);
      if (NT > 1) {
        b[NT - 1][0] = ldg_keep(x1 + ch * 8 + 2 * c);
        b[NT - 1][1] = ldg_keep(x1 + ch * 8 + 2 * c + 1);
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

#define GEMM_ARGS const u16* x, const int* tok, const int* blk_exp, const int* blk_row, \
  const int* blk_cnt, const int* nblk, const uint4* w, const uint4* sf, const float* gs, \
  const u16* nw, void* out, int R, int local_start, int K, int N, int split
#define GEMM_PASS x, tok, blk_exp, blk_row, blk_cnt, nblk, w, sf, gs, nw, out, R, local_start, K, N, split

extern "C" __global__ void __launch_bounds__(128, 4) moe_gemm1(GEMM_ARGS) { gemm<true, false>(GEMM_PASS); }
extern "C" __global__ void __launch_bounds__(128, 4) moe_gemm1_exact(GEMM_ARGS) { gemm<true, true>(GEMM_PASS); }
extern "C" __global__ void __launch_bounds__(128, 4) moe_gemm2(GEMM_ARGS) { gemm<false, false>(GEMM_PASS); }
extern "C" __global__ void __launch_bounds__(128, 4) moe_gemm2_exact(GEMM_ARGS) { gemm<false, true>(GEMM_PASS); }
"""


# Where register j's code sits inside each 16-bit half of a word: sign bit and lowest of the
# three exponent/mantissa bits. Chosen so the decode is cheap: j = 0 needs one mask, j = 1 a
# shift and a mask, j = 2 and 3 two shifts and two masks (11 ops per word instead of 15).
CODE_SIGN_BIT = (15, 9, 10, 11)
CODE_MAG_LOW = (6, 0, 3, 12)


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


POST_NORM_EPS = 1e-8
# Tokens per block of one expert (two n8 MMA tiles).
BM = 16
# Route prep by pair count: _route_prep (Triton, one pass of <= 64 pairs) is faster for
# few pairs (2-3 us against ~4 for moe_route on H100); moe_route (CUDA) from there up to
# 1024 threads x 4 pairs; beyond, _route_prep again (pairs per step PREP_CHUNK).
ROUTE_MIN_PAIRS = 64
ROUTE_MAX_PAIRS = 4096
PREP_CHUNK = 256


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
def _swiglu_reduce(part_ptr, inter_ptr, nblk_ptr, R, I: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    """Split-K only: inter[row] = silu(sum_s gate_s) * sum_s up_s, bf16, from
    moe_gemm1's fp32 slices [SPLIT, R, 2I]. Only the live sorted rows (nblk[1], from
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
def _combine(y_ptr, pos_ptr, wts_ptr, out_ptr, eps, R,
             H: tl.constexpr, TOPK: tl.constexpr, KP: tl.constexpr, Y_SPLIT: tl.constexpr):
    """out[t] = sum over t's local pairs of g * rmsnorm(y[pair]); zero if none is local.
    With Y_SPLIT > 1, y[pair] is the sum of moe_gemm2's fp32 slices. All of the token's
    rows are loaded as one [KP, H] tile (KP = TOPK rounded up to a power of 2), so the
    norms take one reduction instead of one per slot."""
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


def _needs_exact(sf: torch.Tensor) -> bool:
    """True if any E4M3 scale is zero, subnormal, negative or NaN: the fast decode
    (0x5800 | (b & 0x7F) << 4) is right only for positive normal finite scales."""
    b = sf.view(torch.uint8)
    return bool((((b & 0x78) == 0) | ((b & 0x80) != 0) | ((b & 0x7F) == 0x7F)).any())


def prepare(weights: Weights):
    """Compiles the CUDA GEMMs (NVRTC, in process) and repacks the NVFP4 codes and E4M3
    scales into their fragment layout (pack_codes, pack_scales: a reorder of bytes, the
    values and their precision untouched). post_norm_w stays BF16; the kernels add the 1.
    Each GEMM takes the exact scale decode only if its scales need it (_needs_exact)."""
    dev = weights["w13"].device
    hid, inter = weights["w13"].shape[2] * 2, weights["w2"].shape[2] * 2
    assert hid % (CHUNK * GROUP) == 0 and inter % (CHUNK * GROUP) == 0, "K must be a multiple of 512"
    assert inter % 16 == 0 and hid % 32 == 0, "GEMM1 tiles 16 columns of I, GEMM2 32 rows of H"
    assert hid & (hid - 1) == 0, "_combine reads a whole row with tl.arange(0, H): H must be a power of 2"
    return {
        "w13_exact": _needs_exact(weights["w13_sf"]),
        "w2_exact": _needs_exact(weights["w2_sf"]),
        "inter": inter,
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2_gs": weights["w2_gs"].float().contiguous(),
        "cuda": _compile_cuda(),
        "w13": _pack_per_expert(weights["w13"]),
        "w13_sf": pack_scales(weights["w13_sf"].view(torch.uint8)),
        "w2": _pack_per_expert(weights["w2"]),
        "w2_sf": pack_scales(weights["w2_sf"].view(torch.uint8)),
        "norm_w": weights["post_norm_w"].contiguous(),
        "num_sms": torch.cuda.get_device_properties(dev).multi_processor_count,
    }


# K slices for 7 <= bs <= 64. Work items are near-uniform (a whole K slice of a weight
# tile), so the last round-robin round is partly idle; halving the items shrinks that
# tail (+4.7% on H200; 4 slices, or 2 also above bs 64, measured the same).
SPLIT_MID = 2


def _split_k(bs: int, k: int) -> int:
    """K slices per work item, by token-count range. With few tokens only ~2-3 experts are
    touched and there are too few items for the GPU's warps; splitting K multiplies the
    items and shortens each warp's chain. A slice is a whole number of 512-k groups, so the
    count must divide k / 512: otherwise some groups would be summed twice."""
    want = 4 if bs <= 2 else 2 if bs <= 6 else SPLIT_MID if bs <= 64 else 1
    groups = k // (CHUNK * GROUP)
    while groups % want:
        want -= 1
    return want


def kernel_fn(data: Inputs, state) -> torch.Tensor:
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

    tok = torch.empty(r, dtype=torch.int32, device=dev)
    pos = torch.empty(r, dtype=torch.int32, device=dev)
    blk = torch.empty((3, max_blocks), dtype=torch.int32, device=dev)
    nblk = torch.empty(2, dtype=torch.int32, device=dev)  # [blocks, live sorted rows]
    if ROUTE_MIN_PAIRS < r <= ROUTE_MAX_PAIRS and n_local <= 256:
        state["cuda"]["moe_route"](grid=(1, 1, 1), block=(1024, 1, 1),
                                   args=[ids, r, start, n_local, top_k, BM, tok, pos, blk[0], blk[1], blk[2], nblk])
    else:
        _route_prep[(1,)](ids, r, start, tok, pos, blk[0], blk[1], blk[2], nblk,
                          N_LOCAL=n_local, BLOCK_E=triton.next_power_of_2(n_local),
                          CHUNK=min(PREP_CHUNK, triton.next_power_of_2(r)),
                          BLOCK_B=triton.next_power_of_2(max_blocks), BM=BM, TOPK=top_k,
                          num_warps=4 if r <= 128 else 8)

    s1 = _split_k(m_tok, hid)
    s2 = _split_k(m_tok, inter_dim)
    inter = torch.empty((r, inter_dim), dtype=torch.bfloat16, device=dev)
    h = inter if s1 == 1 else torch.empty((s1, r, 2 * inter_dim), dtype=torch.float32, device=dev)
    # One warp per item: (block, 16 inter columns, K slice); 4 warps per CTA.
    kern = state["cuda"]["moe_gemm1_exact" if state["w13_exact"] else "moe_gemm1"]
    cg = min(state["num_sms"] * CUDA_CTAS_PER_SM, triton.cdiv(max_blocks * (inter_dim // 16) * s1, 4))
    kern(grid=(cg, 1, 1), block=(128, 1, 1),
         args=[x, tok, blk[0], blk[1], blk[2], nblk, state["w13"], state["w13_sf"], state["w13_gs"],
               state["norm_w"], h, r, start, hid, inter_dim, s1])
    if s1 > 1:
        blk_i = min(1024, inter_dim)
        _swiglu_reduce[(r, inter_dim // blk_i)](h, inter, nblk, r, I=inter_dim, SPLIT=s1, BLOCK=blk_i, num_warps=4)
    y = torch.empty((r, hid) if s2 == 1 else (s2, r, hid), dtype=torch.bfloat16 if s2 == 1 else torch.float32,
                    device=dev)
    # One warp per item: (block, 32 output rows, K slice).
    kern = state["cuda"]["moe_gemm2_exact" if state["w2_exact"] else "moe_gemm2"]
    cg = min(state["num_sms"] * CUDA_CTAS_PER_SM, triton.cdiv(max_blocks * (hid // 32) * s2, 4))
    kern(grid=(cg, 1, 1), block=(128, 1, 1),
         args=[inter, tok, blk[0], blk[1], blk[2], nblk, state["w2"], state["w2_sf"], state["w2_gs"],
               state["norm_w"], y, r, start, inter_dim, hid, s2])
    out = torch.empty((m_tok, hid), dtype=torch.bfloat16, device=dev)
    _combine[(m_tok,)](y, pos, wts, out, POST_NORM_EPS, r, H=hid, TOPK=top_k, KP=triton.next_power_of_2(top_k),
                       Y_SPLIT=s2, num_warps=16 if m_tok <= 64 else 8)
    return out
