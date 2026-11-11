"""Full-attention (and long split) layers in CUDA: kernel A of the split-KV path with an explicit
cp.async pipeline, FlashAttention-2 style math in registers. Writes the same per-split partials
(normalized o, base-2 log-sum-exp) as the Triton kernel A, so the Triton combine kernel merges
them unchanged.

Why: Nsight Compute on the Triton kernel A (verify 16 full) shows each key block paying, in
series, the page-table load (the address math waits on it) and then the key/value copies (a
barrier right after the cp.async wait): Triton's pipeliner cannot overlap a gather whose address
comes from another load. Here the pipeline is explicit:

- a ring of STAGES raw E4M3 key/value tiles in shared memory, filled with cp.async;
- the page-table entries of block j fetched (cp.async too) 2 * STAGES - 1 blocks ahead, so each
  block's row addresses are in shared memory before its copies are issued;
- one cp.async group per block, `cp.async.wait_group(STAGES - 2)` and one barrier per iteration;
- QK^T in E4M3 (`mma.sync m16n8k32`, sm89+): q rounded to E4M3 at unit RMS as the Triton
  kernel's QK_FP8, the scale on the scores; K's B fragments are the raw tile bytes (16-byte
  loads, the reduction dims permuted the same way in Q);
- P V in FP16: V's B fragments from 8-byte loads of two rows (output dims permuted inside each
  head), byte permutes and one cvt per n-tile. Rows padded 16 bytes past the data: no bank
  conflicts. No FP16 copy in shared memory.

Warps: decode (16 block-diagonal rows for a KV pair) splits each 64-key block's keys over the 4
warps and merges them at the end in shared memory; verify (64 rows per KV head) gives each warp
16 rows and all the block's keys."""

import ctypes
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sliding_cuda as _infra  # noqa: E402  (NVRTC helpers)

SOURCE = r'''
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>

// -D: HQ HKV L KVP WARP_M WARP_K STAGES PAGE WINDOW ROPE MINB (CTAs per SM the registers must fit)
#define D 64
#define BN 64                          // keys per block
#define G (HQ / HKV)
#define QKV_STRIDE ((2 * HQ + 2 * HKV) * D)
#define ROW (HKV * D)
#define NWARPS (WARP_M * WARP_K)
#define NT (NWARPS * 32)
#define DIMS (KVP * D)
#define QP (DIMS + 16)                 // Q rows (E4M3 bytes), padded
#define RP (DIMS + 16)                 // raw E4M3 rows (bytes), padded
#define ROWS (WARP_M * 16)
#define NREAL (KVP * L * G)
#define PRING (2 * STAGES)             // page-table ring slots
#define NEG_INF __int_as_float(0xff800000)
// CL > 0: the whole step in one kernel. A cluster of CL CTAs per (request, KV group) takes the CL
// splits; each keeps its normalized partial in shared memory, and after a cluster barrier every CTA
// merges a few output rows through DSMEM, then the norm-gate and the bfloat16 store (no combine).
#ifndef LASTM
#define LASTM 0                        // 1 (decode, no cluster): the last CTA of a (request, KV pair) merges
#endif
#ifndef GDCQ
#define GDCQ 0                         // 1 (inside a layer, launched with PDL): griddepcontrol.wait after the
                                       // K/V chain is issued and before any read of qkv or write of the cache
#endif
#ifndef EPIW
#define EPIW 0                         // 1: the partials' weights once per row (epilogue)
#endif
#ifndef KVEARLY
#define KVEARLY 0                      // 1: the first blocks' K/V copies before the queries are done
#endif
#ifndef CL
#define CL 0
#endif
// NEWSMEM 1: the split holding block 0 also writes the step's new keys and values (E4M3) straight
// into its stage-0 tile and skips their copies, so it neither fences nor reads them back.
#ifndef NEWSMEM
#define NEWSMEM 0
#endif
// TSTAMP (diagnostic): thread 0 of each CTA stores %globaltimer at each phase into ws_o (as u64,
// [CTA][8]).
// EARLYQ 1: the queries' loads go first (they don't depend on seq_lens), then the page entries,
// and the block-0 split writes the step's keys and values last, overlapped with the rest.
#ifndef EARLYQ
#define EARLYQ 0
#endif
#ifndef TSTAMP
#define TSTAMP 0
#endif
#define TS(k) do { if (TSTAMP && tid == 0) { unsigned long long t_; \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); \
    const unsigned cta_ = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x; \
    ((unsigned long long*)ws_o)[cta_ * 8 + (k)] = t_; } } while (0)
// Memory-path experiments (defaults = the submitted kernel): KVORD 1 issues a block's K rows before
// its V rows; CPCA 1 copies through L1 (cp.async.ca); GSWAP 1 puts the KV group on blockIdx.x.
#ifndef KVORD
#define KVORD 0
#endif
#ifndef CPCA
#define CPCA 0
#endif
#ifndef GSWAP
#define GSWAP 0
#endif

#define OFF_Q 0
#define OFF_K (OFF_Q + ROWS * QP)
#define OFF_V (OFF_K + STAGES * BN * RP)
#define OFF_P (OFF_V + STAGES * BN * RP)
// CL: the partials every rank sends to the owner of each merge row (rho on rank rho % CL, slot
// rho / CL): [MROWS][CL][D + 1] floats (o, then the lse), apart from everything the main loop uses.
#define MROWS (CL ? (KVP * L * G + CL - 1) / (CL ? CL : 1) : 0)
#define OFF_MG (OFF_P + PRING * BN * 4)
#define SMEM_BYTES (OFF_MG + MROWS * (CL ? CL : 1) * (D + 1) * 4)

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void mma_k8(float* d, unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *(unsigned*)&h;
}
__device__ __forceinline__ unsigned e4m3x2_to_h2(unsigned short v) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
    return *(unsigned*)&h;
}
// D += A B over k = 32, A and B in E4M3 (sm89+). Below sm89 (local logic tests), four k = 8 FP16
// MMAs over the same registers: slice s takes bytes 2 (s & 1), + 1 of registers a[(s >> 1) 2],
// a[(s >> 1) 2 + 1] and b[s >> 1], the same k values in A and B, so the sum is the same.
__device__ __forceinline__ void mma_k32_e4m3(float* d, const unsigned* a, const unsigned* b) {
#if __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
#pragma unroll
    for (int sl = 0; sl < 4; ++sl) {
        const int r = (sl >> 1) * 2, sh = 16 * (sl & 1);
        mma_k8(d, e4m3x2_to_h2((unsigned short)(a[r] >> sh)), e4m3x2_to_h2((unsigned short)(a[r + 1] >> sh)),
               e4m3x2_to_h2((unsigned short)(b[sl >> 1] >> sh)));
    }
#endif
}
// cp.async: 16 and 4 bytes, global to shared (sm80+; plain copies below, for local logic tests).
// cp.async goes through the generic proxy, so the __syncthreads after cp.async.wait_group is
// enough; a TMA (async proxy) version would also need fence.proxy.async before reusing a stage.
__device__ __forceinline__ void cp16(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
#if CPCA
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr(dst)), "l"(src));
#else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr(dst)), "l"(src));
#endif
#else
    *(uint4*)dst = *(const uint4*)src;
#endif
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr(dst)), "l"(src));
#else
    *(int*)dst = *(const int*)src;
#endif
}
__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> __device__ __forceinline__ void cp_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
__device__ __forceinline__ float2 norm_rope(float2 x, float2 cs, float eps) {
    float inv = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / D + eps);
    x.x *= inv; x.y *= inv;
    return make_float2(x.x * cs.x - x.y * cs.y, x.y * cs.x + x.x * cs.y);
}

extern "C" __global__ void __launch_bounds__(NT, MINB) full_fa2(
    const __nv_bfloat16* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const int* __restrict__ seq_lens, __nv_fp8_e4m3* k_cache, __nv_fp8_e4m3* v_cache,
    const float* __restrict__ k_scale_ptr, const float* __restrict__ v_scale_ptr,
    const int* __restrict__ page_table, const float* __restrict__ rope,
    float* __restrict__ ws_o, float* __restrict__ ws_lse, int pt_stride, int num_splits, float q_mult, float eps,
    __nv_bfloat16* __restrict__ out, int* __restrict__ cnt)
{
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* Qs = smem + OFF_Q;   // [ROWS][QP] E4M3, dims permuted (below)
    unsigned char* Kr = smem + OFF_K;   // [STAGES][BN][RP] E4M3
    unsigned char* Vr = smem + OFF_V;
    int* Pg = (int*)(smem + OFF_P);     // [PRING][BN] page-table entries

    const int split = GSWAP ? blockIdx.y : blockIdx.x, kv0 = (GSWAP ? blockIdx.x : blockIdx.y) * KVP, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, grp = lane >> 2, tig = lane & 3;
    TS(0);
#if EARLYQ
    // Queries meanwhile: RMSNorm, RoPE, E4M3; decode packs the pair block-diagonally.
    // All of a warp's row loads go out before any is used (one latency, not one per row).
    constexpr int QR = (ROWS + NWARPS - 1) / NWARPS;
    float2 qx[QR], qcs[QR];
    long qpos[QR];
#pragma unroll
    for (int i = 0; i < QR; ++i) {
        int r = warp + i * NWARPS, h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        qx[i] = make_float2(0.f, 0.f);
        qpos[i] = 0;
        if (r < ROWS && rr < L * G) {
            int j = rr / G, g = rr % G, tok = b * L + j;
            const __nv_bfloat162* src =
                (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE + ((kv0 + h) * G + g) * D);
            if (!GDCQ) qx[i] = __bfloat1622float2(src[lane]);   // GDCQ: after the wait, below
            qpos[i] = positions[tok];
        }
    }
    // The step's own keys and values (written by the block-0 split only, but loaded by every CTA:
    // seq_lens isn't known yet), so their chain runs beside the queries' instead of after it.
    constexpr int KR = (KVP * L + NWARPS - 1) / NWARPS;
    float2 kxs[KR], vxs[KR], kcs[KR];
    long kpos[KR], kloc[KR];
#pragma unroll
    for (int i = 0; i < KR; ++i) {
        const int w = warp + i * NWARPS, h = w / L, j = w % L, tok = b * L + j, kvh = kv0 + h;
        kxs[i] = vxs[i] = make_float2(0.f, 0.f);
        kpos[i] = kloc[i] = 0;
        if (w < KVP * L) {
            const __nv_bfloat162* row = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE);
            if (!GDCQ) {
                kxs[i] = __bfloat1622float2(row[(HQ * D + kvh * D) / 2 + lane]);
                vxs[i] = __bfloat1622float2(row[((HQ + HKV) * D + kvh * D) / 2 + lane]);
            }
            kpos[i] = positions[tok];
            kloc[i] = cache_loc[tok];
        }
    }
#endif
    const float k_scale = *k_scale_ptr, v_scale = *v_scale_ptr;
    const int seq = seq_lens[b], hi = seq + L;
    const int lo = WINDOW > 0 ? max(seq - WINDOW + 1, 0) : 0;
    // Blocks counted back from hi: block i holds keys [hi - (i + 1) BN, hi - i BN), clipped at lo.
    // Split s takes blocks [n - (s + 1) per, n - s per), clipped at 0 (as the Triton kernel).
    const int n_blocks = (hi - lo + BN - 1) / BN;
    const int per = (n_blocks + num_splits - 1) / num_splits;
    const int last = n_blocks - split * per, first = max(last - per, 0);
    const int nb = max(last - first, 0);

#if CL && __CUDA_ARCH__ >= 900
    // Every rank has started before any writes into its shared memory: arrive now, wait before the
    // first remote store (by then long satisfied).
    cooperative_groups::this_cluster().barrier_arrive();
#endif
#if CL
    // Gates of this warp's merge rows (see the merge below; the rank is the split), loaded now: the
    // merge needs them last, after a DRAM latency nobody waits on.
    constexpr int GR = (KVP * L * G + CL * NWARPS - 1) / (CL * NWARPS);
    float2 gate_r[GR];
#pragma unroll
    for (int i = 0; i < GR; ++i) {
        const int rho = split + CL * (warp + NWARPS * i);
        gate_r[i] = make_float2(0.f, 0.f);
        if (rho < KVP * L * G) {
            const int h = rho / (L * G), rr = rho % (L * G), j = rr / G, head = (kv0 + h) * G + rr % G;
            const __nv_bfloat16* gp = qkv + (long)(b * L + j) * QKV_STRIDE + (HQ + 2 * HKV) * D + head * D;
            if (!GDCQ) gate_r[i] = make_float2(__bfloat162float(gp[lane]), __bfloat162float(gp[lane + 32]));
        }
    }
#endif
#if LASTM
    // Gates of the rows this CTA may merge (any CTA can be the last), loaded now: a warp per row.
    constexpr int LGR = (KVP * L * G + NWARPS - 1) / NWARPS;
    float2 gate_l[LGR];
#pragma unroll
    for (int i = 0; i < LGR; ++i) {
        const int rho = warp + NWARPS * i;
        gate_l[i] = make_float2(0.f, 0.f);
        if (rho < KVP * L * G) {
            const int rr = rho % (L * G), head = (kv0 + rho / (L * G)) * G + rr % G;
            const __nv_bfloat16* gp = qkv + (long)(b * L + rr / G) * QKV_STRIDE + (HQ + 2 * HKV) * D + head * D;
            gate_l[i] = make_float2(__bfloat162float(gp[lane]), __bfloat162float(gp[lane + 32]));
        }
    }
#endif
#if !EARLYQ
    // The split holding block 0 writes the step's new keys and values to the cache first; its own
    // copies of block 0 then read them back (after the fence and barrier below). NEWSMEM: it puts
    // them in its stage-0 tile too (rows BN - L + j of block 0) and copies none of them.
    if (first == 0 && last > 0) {
        for (int w = warp; w < KVP * L; w += NWARPS) {
            int h = w / L, j = w % L, tok = b * L + j, kvh = kv0 + h;
            long pos = positions[tok], loc = cache_loc[tok];
            const __nv_bfloat162* row = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE);
            float2 kx = __bfloat1622float2(row[(HQ * D + kvh * D) / 2 + lane]);
            float2 vx = __bfloat1622float2(row[((HQ + HKV) * D + kvh * D) / 2 + lane]);
            float2 cs = ROPE ? make_float2(rope[pos * D + lane], rope[pos * D + D / 2 + lane])
                             : make_float2(1.f, 0.f);
            kx = norm_rope(kx, cs, eps);
            __nv_fp8x2_storage_t k8 = __nv_cvt_float2_to_fp8x2(make_float2(kx.x / k_scale, kx.y / k_scale),
                                                                __NV_SATFINITE, __NV_E4M3);
            __nv_fp8x2_storage_t v8 = __nv_cvt_float2_to_fp8x2(make_float2(vx.x / v_scale, vx.y / v_scale),
                                                                __NV_SATFINITE, __NV_E4M3);
            *(__nv_fp8x2_storage_t*)(k_cache + loc * ROW + kvh * D + 2 * lane) = k8;
            *(__nv_fp8x2_storage_t*)(v_cache + loc * ROW + kvh * D + 2 * lane) = v8;
            if (NEWSMEM) {
                *(__nv_fp8x2_storage_t*)(smem + OFF_K + (BN - L + j) * RP + h * D + 2 * lane) = k8;
                *(__nv_fp8x2_storage_t*)(smem + OFF_V + (BN - L + j) * RP + h * D + 2 * lane) = v8;
            }
        }
        if (!NEWSMEM) __threadfence();
    }
#endif

    // Pipeline helpers: block number j (0 .. nb) of this split -> block index first + j.
    auto issue_pages = [&](int j) {   // page entries of block j into ring slot j % PRING
        if (j < nb) {
            for (int key = tid; key < BN; key += NT) {
                int t = hi - (first + j + 1) * BN + key;
                if (t >= lo) cp4(Pg + (j % PRING) * BN + key, page_table + (long)b * pt_stride + t / PAGE);
            }
        }
    };
    auto issue_kv = [&](int j) {      // raw K and V rows of block j into stage j % STAGES
        constexpr int CPK = DIMS / 16;
        if (j < nb) {
            for (int idx = tid; idx < (KVORD ? 2 : 1) * BN * CPK; idx += NT) {
                const int kv = KVORD ? idx / (BN * CPK) : 0, i = KVORD ? idx % (BN * CPK) : idx;
                int key = i / CPK, c16 = (i % CPK) * 16, t = hi - (first + j + 1) * BN + key;
                unsigned char* dk = Kr + ((j % STAGES) * BN + key) * RP + c16;
                unsigned char* dv = Vr + ((j % STAGES) * BN + key) * RP + c16;
                if (t < lo) {  // clipped rows: zeros (stale bytes could encode NaN, and 0 * NaN = NaN)
                    if (!KVORD || kv == 0) *(uint4*)dk = make_uint4(0, 0, 0, 0);
                    if (!KVORD || kv == 1) *(uint4*)dv = make_uint4(0, 0, 0, 0);
                    continue;
                }
                if (NEWSMEM && first + j == 0 && t >= seq) continue;   // the step's own rows: written above
                long row = (long)Pg[(j % PRING) * BN + key] * PAGE + t % PAGE;
                long at = row * ROW + kv0 * D + c16;
                if (!KVORD || kv == 0) cp16(dk, k_cache + at);
                if (!KVORD || kv == 1) cp16(dv, v_cache + at);
            }
        }
    };

    // Prologue: page entries of blocks 0 .. 2 STAGES - 2 (one wait), then the copies of blocks
    // 0 .. STAGES - 2, one group each.
    for (int j = 0; j < 2 * STAGES - 1; ++j) issue_pages(j);
    cp_commit();
#if !EARLYQ
    // Queries meanwhile: RMSNorm, RoPE, E4M3; decode packs the pair block-diagonally.
    // All of a warp's row loads go out before any is used (one latency, not one per row).
    constexpr int QR = (ROWS + NWARPS - 1) / NWARPS;
    float2 qx[QR], qcs[QR];
    long qpos[QR];
#pragma unroll
    for (int i = 0; i < QR; ++i) {
        int r = warp + i * NWARPS, h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        qx[i] = make_float2(0.f, 0.f);
        qpos[i] = 0;
        if (r < ROWS && rr < L * G) {
            int j = rr / G, g = rr % G, tok = b * L + j;
            const __nv_bfloat162* src =
                (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE + ((kv0 + h) * G + g) * D);
            qx[i] = __bfloat1622float2(src[lane]);
            qpos[i] = positions[tok];
        }
    }
#endif
#pragma unroll
    for (int i = 0; i < QR; ++i)
        qcs[i] = ROPE ? make_float2(rope[qpos[i] * D + lane], rope[qpos[i] * D + D / 2 + lane])
                      : make_float2(1.f, 0.f);
#if KVEARLY
    // The first blocks' copies as soon as their page entries land, while the RoPE loads above are in
    // flight: the K/V chain (seq_lens -> pages -> rows) no longer waits for the queries' chain.
    cp_wait<0>();
    __syncthreads();
    for (int j = 0; j < STAGES - 1; ++j) { issue_kv(j); cp_commit(); }
#endif
#if GDCQ
    static_assert(!GDCQ || (EARLYQ && NEWSMEM && KVEARLY), "GDCQ: the step's own rows from shared memory, K/V first");
    // Everything before this line reads only what the layer's earlier kernels don't write (seq_lens,
    // the page table, positions, cache_loc, the RoPE table, the cache's older rows; NEWSMEM skips the
    // step's own rows). Now the qkv producer's results: q, the new k/v (and the gate).
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
#pragma unroll
    for (int i = 0; i < QR; ++i) {
        int r = warp + i * NWARPS, h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        if (r < ROWS && rr < L * G) {
            int j = rr / G, g = rr % G, tok = b * L + j;
            const __nv_bfloat162* src = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE + ((kv0 + h) * G + g) * D);
            qx[i] = __bfloat1622float2(src[lane]);
        }
    }
#pragma unroll
    for (int i = 0; i < KR; ++i) {
        const int w = warp + i * NWARPS, h = w / L, j = w % L, tok = b * L + j, kvh = kv0 + h;
        if (w < KVP * L) {
            const __nv_bfloat162* row = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE);
            kxs[i] = __bfloat1622float2(row[(HQ * D + kvh * D) / 2 + lane]);
            vxs[i] = __bfloat1622float2(row[((HQ + HKV) * D + kvh * D) / 2 + lane]);
        }
    }
#if CL
#pragma unroll
    for (int i = 0; i < GR; ++i) {
        const int rho = split + CL * (warp + NWARPS * i);
        if (rho < KVP * L * G) {
            const int h = rho / (L * G), rr = rho % (L * G), j = rr / G, head = (kv0 + h) * G + rr % G;
            const __nv_bfloat16* gp = qkv + (long)(b * L + j) * QKV_STRIDE + (HQ + 2 * HKV) * D + head * D;
            gate_r[i] = make_float2(__bfloat162float(gp[lane]), __bfloat162float(gp[lane + 32]));
        }
    }
#endif
#endif
#if EARLYQ
    static_assert(!EARLYQ || NEWSMEM, "EARLYQ: the block-0 split reads its own rows from shared memory");
#pragma unroll
    for (int i = 0; i < KR; ++i)
        kcs[i] = ROPE ? make_float2(rope[kpos[i] * D + lane], rope[kpos[i] * D + D / 2 + lane]) : make_float2(1.f, 0.f);
    if (first == 0 && last > 0) {
#pragma unroll
        for (int i = 0; i < KR; ++i) {
            const int w = warp + i * NWARPS, h = w / L, j = w % L, kvh = kv0 + h;
            if (w >= KVP * L) continue;
            const float2 kx = norm_rope(kxs[i], kcs[i], eps), vx = vxs[i];
            const __nv_fp8x2_storage_t k8 = __nv_cvt_float2_to_fp8x2(make_float2(kx.x / k_scale, kx.y / k_scale),
                                                                      __NV_SATFINITE, __NV_E4M3);
            const __nv_fp8x2_storage_t v8 = __nv_cvt_float2_to_fp8x2(make_float2(vx.x / v_scale, vx.y / v_scale),
                                                                      __NV_SATFINITE, __NV_E4M3);
            *(__nv_fp8x2_storage_t*)(k_cache + kloc[i] * ROW + kvh * D + 2 * lane) = k8;
            *(__nv_fp8x2_storage_t*)(v_cache + kloc[i] * ROW + kvh * D + 2 * lane) = v8;
            *(__nv_fp8x2_storage_t*)(smem + OFF_K + (BN - L + j) * RP + h * D + 2 * lane) = k8;
            *(__nv_fp8x2_storage_t*)(smem + OFF_V + (BN - L + j) * RP + h * D + 2 * lane) = v8;
        }
    }
#endif
    const float qsc = q_mult * k_scale;
#pragma unroll
    for (int i = 0; i < QR; ++i) {
        int r = warp + i * NWARPS, h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        if (r >= ROWS) continue;
        float2 x = norm_rope(qx[i], qcs[i], eps);
        if (rr >= L * G) x = make_float2(0.f, 0.f);
        // Unit-RMS q to E4M3 (as the Triton kernel's QK_FP8); the scale goes on the scores. Dims are
        // permuted to match K's reads below: thread tig reads key bytes tig CH .. + CH (CH = DIMS / 4),
        // and byte t 8 + r 4 + i of them is element i of B register r of k-tile t, so dim
        // d = tig CH + t 8 + r 4 + i goes where k-tile t's A fragment puts thread tig: t 32 + r 16 + tig 4 + i.
#pragma unroll
        for (int hh = 0; hh < KVP; ++hh) {
            const int d = hh * D + 2 * lane, rem = d % (DIMS / 4);
            const int at = (rem / 8) * 32 + ((rem / 4) & 1) * 16 + (d / (DIMS / 4)) * 4 + rem % 4;
            *(__nv_fp8x2_storage_t*)(Qs + r * QP + at) =
                hh == h ? __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3) : (__nv_fp8x2_storage_t)0;
        }
    }
#if KVEARLY
    __syncthreads();
    TS(1);   // queries in shared memory (the first blocks' copies already in flight)
#else
    cp_wait<0>();
    __syncthreads();
    TS(1);   // queries in shared memory, the first page entries landed
    for (int j = 0; j < STAGES - 1; ++j) { issue_kv(j); cp_commit(); }
#endif

    const int mt = warp % WARP_M, ks = warp / WARP_M;
    constexpr int KT = DIMS / 32, NO = DIMS / 8;   // k-tiles of QK^T (E4M3, k = 32), n-tiles of O
    constexpr int WKEYS = BN / WARP_K;      // keys of each block per warp
    unsigned qa[KT][4];                     // A fragments: rows grp, grp + 8; bytes tig 4, 16 + tig 4
#pragma unroll
    for (int kt = 0; kt < KT; ++kt)
#pragma unroll
        for (int r = 0; r < 4; ++r)
            qa[kt][r] = *(const unsigned*)(Qs + (mt * 16 + grp + (r & 1) * 8) * QP + kt * 32 + (r >> 1) * 16 + tig * 4);
    int ptok[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        int rr = (mt * 16 + grp + e * 8) % (ROWS / KVP);
        ptok[e] = seq + min(rr / G, L - 1);
    }
    float o[NO][4];
#pragma unroll
    for (int n = 0; n < NO; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m_run[2] = {NEG_INF, NEG_INF}, l_run[2] = {0.f, 0.f};

    for (int j = 0; j < nb; ++j) {
        cp_wait<STAGES - 2>();   // block j's copies have landed (and the pages STAGES - 1 ahead)
        __syncthreads();          // ... for every thread; and nobody still reads the stage refilled next
        if (j == 0) TS(2);        // block 0's keys and values landed
        issue_pages(j + 2 * STAGES - 1);
        issue_kv(j + STAGES - 1);
        cp_commit();

        const unsigned char* K = Kr + (j % STAGES) * BN * RP;
        const unsigned char* V = Vr + (j % STAGES) * BN * RP;
        const int kb0 = hi - (first + j + 1) * BN;     // position of key 0 of this block
        const int k_lo = ks * WKEYS;                   // this warp's keys of the block
        float s[WKEYS / 8][4];
#pragma unroll
        for (int nt = 0; nt < WKEYS / 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
            // B fragments of key grp: bytes tig CH .. + CH (CH = DIMS / 4 = 8 KT), 16-byte loads;
            // k-tile kt takes words 2 kt, 2 kt + 1, raw E4M3 (no conversion).
            const unsigned char* krow = K + (k_lo + nt * 8 + grp) * RP + tig * 8 * KT;
            unsigned kw[2 * KT];
#pragma unroll
            for (int c = 0; c < KT / 2; ++c) {
                const uint4 u = *(const uint4*)(krow + 16 * c);
                kw[4 * c] = u.x; kw[4 * c + 1] = u.y; kw[4 * c + 2] = u.z; kw[4 * c + 3] = u.w;
            }
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) mma_k32_e4m3(s[nt], qa[kt], kw + 2 * kt);
        }
        float mx[2] = {NEG_INF, NEG_INF};
#pragma unroll
        for (int nt = 0; nt < WKEYS / 8; ++nt)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                int e = c >> 1, t = kb0 + k_lo + nt * 8 + tig * 2 + (c & 1), p = ptok[e];
                bool ok = t >= lo && t <= p && (WINDOW == 0 || t > p - WINDOW);
                s[nt][c] = ok ? s[nt][c] * qsc : NEG_INF;
                mx[e] = fmaxf(mx[e], s[nt][c]);
            }
        float alpha[2], safe[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 1));
            mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 2));
            float m_new = fmaxf(m_run[e], mx[e]);
            safe[e] = m_new == NEG_INF ? 0.f : m_new;
            alpha[e] = exp2f(m_run[e] - safe[e]);
            m_run[e] = m_new;
        }
        float rs[2] = {0.f, 0.f};
        unsigned pa[WKEYS / 8][2];
#pragma unroll
        for (int nt = 0; nt < WKEYS / 8; ++nt) {
            float p0 = exp2f(s[nt][0] - safe[0]), p1 = exp2f(s[nt][1] - safe[0]);
            float p2 = exp2f(s[nt][2] - safe[1]), p3 = exp2f(s[nt][3] - safe[1]);
            rs[0] += p0 + p1; rs[1] += p2 + p3;
            pa[nt][0] = pack_h2(p0, p1); pa[nt][1] = pack_h2(p2, p3);
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 1);
            rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 2);
            l_run[e] = l_run[e] * alpha[e] + rs[e];
        }
#pragma unroll
        for (int n = 0; n < NO; ++n) {
            o[n][0] *= alpha[0]; o[n][1] *= alpha[0]; o[n][2] *= alpha[1]; o[n][3] *= alpha[1];
        }
        // O += P V. Output dims are permuted inside each head: column c of n-tile h * 8 + n is dim
        // h * 64 + c * 8 + n. The B fragment (keys 2 tig, 2 tig + 1; column grp) of n-tiles
        // h * 8 .. + 8 is then 8 consecutive bytes of each of two rows: two 8-byte loads, byte
        // permutes pair the rows, one cvt per n-tile.
#pragma unroll
        for (int kt = 0; kt < WKEYS / 8; ++kt) {
            const unsigned char* v0 = V + (k_lo + kt * 8 + 2 * tig) * RP + grp * 8;
#pragma unroll
            for (int h = 0; h < KVP; ++h) {
                const uint2 a = *(const uint2*)(v0 + h * D), c = *(const uint2*)(v0 + RP + h * D);
                const unsigned pr[4] = {__byte_perm(a.x, c.x, 0x5140), __byte_perm(a.x, c.x, 0x7362),
                                        __byte_perm(a.y, c.y, 0x5140), __byte_perm(a.y, c.y, 0x7362)};
#pragma unroll
                for (int n = 0; n < 8; ++n)
                    mma_k8(o[h * 8 + n], pa[kt][0], pa[kt][1],
                           e4m3x2_to_h2((unsigned short)(pr[n >> 1] >> (16 * (n & 1)))));
            }
        }
    }
    cp_wait<0>();
    TS(3);   // main loop done

    // Partials: rows keep their own head's 64 dims (n-tiles h * 8 .. + 8). Key slices (decode)
    // merge through shared memory (the stages are free now). Then normalized o and lse to ws.
    __syncthreads();
    float* Wm = (float*)(smem + OFF_K);     // [NWARPS][16][2]
    float* Wo = Wm + NWARPS * 16 * 2;       // [NWARPS][16][D]
    // Row grp + 8 e of the m-tile belongs to head e on decode (rows 0-7 head 0, 8-15 head 1),
    // to the program's only head on verify: a compile-time index into o. Decode with L G < 8
    // fails here at compile time, so _cuda_kernel_consts keeps the Triton kernel for it (with the
    // "CUDA kernel A unavailable" line; no test or scored shape has L G < 8).
    static_assert(KVP == 1 || (WARP_M == 1 && L * G == 8), "decode packs 8 rows per head");
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int rl = grp + e * 8, h = KVP == 1 ? 0 : e;
        if (tig == 0) { Wm[(warp * 16 + rl) * 2] = m_run[e]; Wm[(warp * 16 + rl) * 2 + 1] = l_run[e]; }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            Wo[(warp * 16 + rl) * D + tig * 16 + n] = o[h * 8 + n][e * 2];       // dim (2 tig) 8 + n
            Wo[(warp * 16 + rl) * D + tig * 16 + 8 + n] = o[h * 8 + n][e * 2 + 1];
        }
    }
    __syncthreads();
#if EPIW
    // Each row's slice weights (exp2 of the max gap, over the row's denominator, times v_scale) and
    // its lse, once per row instead of once per (row, dim).
    float* Ww = Wo + NWARPS * 16 * D;       // [ROWS][WARP_K + 1]
    if (tid < ROWS) {
        const int mr = tid / 16, rl = tid % 16;
        float mm = NEG_INF, wgt[WARP_K], den = 0.f;
#pragma unroll
        for (int k = 0; k < WARP_K; ++k) mm = fmaxf(mm, Wm[((k * WARP_M + mr) * 16 + rl) * 2]);
        const float safe = mm == NEG_INF ? 0.f : mm;
#pragma unroll
        for (int k = 0; k < WARP_K; ++k) {
            const int w = k * WARP_M + mr;
            wgt[k] = exp2f(Wm[(w * 16 + rl) * 2] - safe);
            den += wgt[k] * Wm[(w * 16 + rl) * 2 + 1];
        }
#pragma unroll
        for (int k = 0; k < WARP_K; ++k) Ww[tid * (WARP_K + 1) + k] = den > 0.f ? wgt[k] / den * v_scale : 0.f;
        Ww[tid * (WARP_K + 1) + WARP_K] = den > 0.f ? mm + __log2f(den) : NEG_INF;
    }
    __syncthreads();
#endif
#if CL && __CUDA_ARCH__ >= 900
    cooperative_groups::this_cluster().barrier_wait();   // every rank has started (arrived at the top)
#endif
    for (int idx = tid; idx < ROWS * D; idx += NT) {
        int r = idx / D, d = idx % D, mr = r / 16, rl = r % 16;
        int h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        if (rr >= L * G) continue;
#if EPIW
        float val = 0.f;
#pragma unroll
        for (int k = 0; k < WARP_K; ++k) val += Ww[r * (WARP_K + 1) + k] * Wo[((k * WARP_M + mr) * 16 + rl) * D + d];
        const float lse = Ww[r * (WARP_K + 1) + WARP_K];
#else
        float mm = NEG_INF;
        for (int k = 0; k < WARP_K; ++k) mm = fmaxf(mm, Wm[((k * WARP_M + mr) * 16 + rl) * 2]);
        float safe = mm == NEG_INF ? 0.f : mm, den = 0.f, acc = 0.f;
        for (int k = 0; k < WARP_K; ++k) {
            int w = k * WARP_M + mr;
            float wgt = exp2f(Wm[(w * 16 + rl) * 2] - safe);
            den += wgt * Wm[(w * 16 + rl) * 2 + 1];
            acc += wgt * Wo[(w * 16 + rl) * D + d];
        }
        const float val = den > 0.f ? acc / den * v_scale : 0.f, lse = den > 0.f ? mm + __log2f(den) : NEG_INF;
#endif
#if CL
        // Row r's partial straight into its owner's buffer (rank rho % CL, slot rho / CL).
        {
            const int rho = h * (L * G) + rr;
#if __CUDA_ARCH__ >= 900
            float* mg = cooperative_groups::this_cluster().map_shared_rank((float*)(smem + OFF_MG), rho % CL);
#else
            float* mg = (float*)(smem + OFF_MG);
#endif
            float* slot = mg + ((rho / CL) * CL + split) * (D + 1);
            slot[d] = val;
            if (d == 0) slot[D] = lse;
        }
#else
        int j = rr / G, g = rr % G, head = (kv0 + h) * G + g;
        long slot = ((long)(b * L + j) * HQ + head) * num_splits + split;
        ws_o[slot * D + d] = val;
        if (d == 0) ws_lse[slot] = lse;
#endif
    }
#if LASTM
    static_assert(!LASTM || (!CL && WARP_M == 1), "LASTM: decode tiles, no cluster");
    // The last CTA of this (request, KV pair) to finish merges every split's partials (from L2), then
    // the norm-gate and the bfloat16 store, as the combine kernel; it resets the counter for the next
    // call (every split has counted by then).
    __threadfence();   // this CTA's partials, before its count
    __syncthreads();
    int* last_flag = (int*)(smem + OFF_K);   // Wm, unused from here on
    if (tid == 0) {
        int* c = cnt + b * gridDim.y + blockIdx.y;
        const int lastc = atomicAdd(c, 1) == num_splits - 1;
        if (lastc) *c = 0;
        *last_flag = lastc;
    }
    __syncthreads();
    if (!*last_flag) return;
    __threadfence();   // the other CTAs' partials, after their counts
    // Every load independent, all issued before use: the lse table, then each row's split weights
    // (one thread per row), then 4 dims per thread with the splits unrolled (16 L2 loads in flight).
    constexpr int R = KVP * L * G;   // 16 rows: 8 per head
    float* Lw = (float*)(smem + OFF_K) + 4;   // [R][num_splits] lse, then weights
    float* Ov = Lw + R * 64;                  // [R][D] merged rows (num_splits <= 64)
    for (int i = tid; i < R * num_splits; i += NT) {
        const int rho = i / num_splits, c = i % num_splits, rr = rho % (L * G);
        const long tok = b * L + rr / G, head = (kv0 + rho / (L * G)) * G + rr % G;
        Lw[i] = __ldcg(ws_lse + (tok * HQ + head) * num_splits + c);
    }
    __syncthreads();
    if (tid < R) {
        float mm = NEG_INF, den = 0.f;
        for (int c = 0; c < num_splits; ++c) mm = fmaxf(mm, Lw[tid * num_splits + c]);
        const float safe = mm == NEG_INF ? 0.f : mm;
        for (int c = 0; c < num_splits; ++c) {   // empty splits: lse -inf, weight 0
            const float w = exp2f(Lw[tid * num_splits + c] - safe);
            Lw[tid * num_splits + c] = w;
            den += w;
        }
        for (int c = 0; c < num_splits; ++c) Lw[tid * num_splits + c] /= den;
    }
    __syncthreads();
    for (int e = tid; e < R * (D / 4); e += NT) {
        const int rho = e / (D / 4), q = e % (D / 4), rr = rho % (L * G);
        const long tok = b * L + rr / G, head = (kv0 + rho / (L * G)) * G + rr % G;
        const float4* src = (const float4*)(ws_o + (tok * HQ + head) * num_splits * D) + q;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int c0 = 0; c0 < num_splits; c0 += 16) {
            float4 v[16];
#pragma unroll
            for (int c = 0; c < 16; ++c)
                v[c] = c0 + c < num_splits ? __ldcg(src + (c0 + c) * (D / 4)) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const float w = c0 + c < num_splits ? Lw[rho * num_splits + c0 + c] : 0.f;
                acc.x += w * v[c].x; acc.y += w * v[c].y; acc.z += w * v[c].z; acc.w += w * v[c].w;
            }
        }
        *(float4*)(Ov + rho * D + 4 * q) = acc;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < LGR; ++i) {   // norm-gate and the bfloat16 store, a warp per row
        const int rho = warp + NWARPS * i;
        if (rho >= R) break;
        const int rr = rho % (L * G), j = rr / G, head = (kv0 + rho / (L * G)) * G + rr % G;
        const long tok = b * L + j;
        const float o0 = Ov[rho * D + lane], o1 = Ov[rho * D + lane + 32];
        const float inv = rsqrtf(warp_sum(o0 * o0 + o1 * o1) / D + eps);
        const float g0 = 1.f / (1.f + __expf(-gate_l[i].x)), g1 = 1.f / (1.f + __expf(-gate_l[i].y));
        out[tok * (HQ * D) + head * D + lane] = __float2bfloat16(o0 * inv * g0);
        out[tok * (HQ * D) + head * D + lane + 32] = __float2bfloat16(o1 * inv * g1);
    }
#endif
#if CL
    // Merge: real row rho on rank rho % CL, warp (rho / CL) % NWARPS; lane holds dims lane, lane + 32.
    const int rank = split;   // the cluster spans the grid's x
#if __CUDA_ARCH__ >= 900
    TS(4);   // partials sent
    cooperative_groups::this_cluster().sync();   // every rank's partials are in their owners' buffers
    TS(5);
#else
    __syncthreads();
#endif
    static_assert(__CUDA_ARCH__ >= 900 || CL == 1, "clusters need sm90");
    static_assert(!CL || !GSWAP, "cl: the cluster rank is the split, blockIdx.x");
#pragma unroll
    for (int i = 0; i < GR; ++i) {
        const int rho = rank + CL * (warp + NWARPS * i);
        if (rho >= KVP * L * G) break;
        const int rr = rho % (L * G), j = rr / G, head = (kv0 + rho / (L * G)) * G + rr % G;
        const float* mg = (const float*)(smem + OFF_MG) + (rho / CL) * CL * (D + 1);
        float mm = NEG_INF;
#pragma unroll
        for (int c = 0; c < CL; ++c) mm = fmaxf(mm, mg[c * (D + 1) + D]);
        const float safe = mm == NEG_INF ? 0.f : mm;
        float den = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int c = 0; c < CL; ++c) {   // empty splits: lse -inf, weight 0
            const float w = exp2f(mg[c * (D + 1) + D] - safe);
            den += w;
            o0 += w * mg[c * (D + 1) + lane];
            o1 += w * mg[c * (D + 1) + lane + 32];
        }
        o0 /= den; o1 /= den;
        const float inv = rsqrtf(warp_sum(o0 * o0 + o1 * o1) / D + eps);
        const long tok = b * L + j;
        const float g0 = 1.f / (1.f + __expf(-gate_r[i].x)), g1 = 1.f / (1.f + __expf(-gate_r[i].y));
        out[tok * (HQ * D) + head * D + lane] = __float2bfloat16(o0 * inv * g0);
        out[tok * (HQ * D) + head * D + lane + 32] = __float2bfloat16(o1 * inv * g1);
    }
    TS(6);
#endif
}
'''

_FUNCS = {}
EXTRA = {}  # extra -D constants (submission.CUDA_EXTRA)
_COUNTERS = {}


def lastm_counters(device):
    """LASTM: a counter per (request, KV pair), zero between calls (the last CTA resets its own)."""
    if device not in _COUNTERS:
        _COUNTERS[device] = torch.zeros(1 << 16, dtype=torch.int32, device=device)
    return _COUNTERS[device]


def consts(cfg, stages=3):
    hq, hkv, length = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["q_len"]
    g = hq // hkv
    decode = hkv % 2 == 0 and 2 * length * g <= 16
    warp_m = 1 if decode else -(-length * g // 16)
    # Verify: cap the registers at 128 for 128-thread CTAs (4 per SM; the V loads cost 7 more).
    return dict(HQ=hq, HKV=hkv, L=length, KVP=2 if decode else 1, WARP_M=warp_m,
                WARP_K=4 if decode else 1, STAGES=stages, PAGE=cfg["page_size"], WINDOW=cfg["window"],
                ROPE=int(bool(cfg["rope_theta"])), MINB=1 if decode else max(1, 16 // warp_m), **EXTRA)


def smem_bytes(k):
    rows, dims = k["WARP_M"] * 16, k["KVP"] * 64
    total = rows * (dims + 16) + 2 * k["STAGES"] * 64 * (dims + 16) + 2 * k["STAGES"] * 64 * 4
    nw = k["WARP_M"] * k["WARP_K"]
    assert nw * 16 * 2 * 4 + nw * 16 * 64 * 4 + rows * 4 <= 2 * k["STAGES"] * 64 * (dims + 16), k  # merge buffers fit
    if k.get("CL"):   # the merge rows' buffers: [MROWS][CL][D + 1] floats
        nreal = k["KVP"] * k["L"] * (k["HQ"] // k["HKV"])
        total += -(-nreal // k["CL"]) * k["CL"] * 65 * 4
    return total


def compile_kernel(k, arch):
    from cuda.bindings import driver, nvrtc
    key = (tuple(sorted(k.items())), arch)
    if key in _FUNCS:
        return _FUNCS[key]
    prog = _infra._cuda_check(nvrtc.nvrtcCreateProgram(SOURCE.encode(), b"full_fa2.cu", 0, [], []))
    opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++17", b"-default-device", b"--use_fast_math"]
    opts += [f"-D{n}={v}".encode() for n, v in k.items()]
    opts += [b"-I" + d.encode() for d in _infra._cuda_include_dirs()]
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    n = _infra._cuda_check(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * n
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(err) != 0:
        raise RuntimeError(f"NVRTC: {log.decode(errors='replace')}")
    n = _infra._cuda_check(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * n
    _infra._cuda_check(nvrtc.nvrtcGetCUBIN(prog, cubin))
    smem = smem_bytes(k)
    if not torch.cuda.is_available():
        _FUNCS[key] = (cubin, smem)
        return _FUNCS[key]
    torch.cuda.init()
    module = _infra._cuda_check(driver.cuModuleLoadData(cubin))
    func = _infra._cuda_check(driver.cuModuleGetFunction(module, b"full_fa2"))
    A = driver.CUfunction_attribute
    _infra._cuda_check(driver.cuFuncSetAttribute(func, A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
    if k.get("CL", 0) > 8:
        _infra._cuda_check(driver.cuFuncSetAttribute(func, A.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1))
    _FUNCS[key] = (func, smem)
    return _FUNCS[key]


def launch(data, ws_o, ws_lse, splits, rope_table, q_mult, eps, stages=3, out=None):
    """Kernel A of the split-KV path: per-split partials into ws_o [tokens, Hq, splits, 64] and
    ws_lse [tokens, Hq, splits], as the Triton kernel A writes them."""
    from cuda.bindings import driver
    cfg = data["config"]
    major, minor = torch.cuda.get_device_capability(ws_o.device)
    k = consts(cfg, stages)
    arch = "sm_90a" if major == 9 else f"sm_{major}{minor}"
    func, smem = compile_kernel(k, arch)
    lc = driver.CUlaunchConfig()
    lc.gridDimX, lc.gridDimY, lc.gridDimZ = splits, cfg["num_kv_heads"] // k["KVP"], cfg["batch"]
    if k.get("GSWAP"):
        lc.gridDimX, lc.gridDimY = lc.gridDimY, lc.gridDimX
    lc.blockDimX, lc.blockDimY, lc.blockDimZ = k["WARP_M"] * k["WARP_K"] * 32, 1, 1
    lc.sharedMemBytes = smem
    lc.hStream = torch.cuda.current_stream(ws_o.device).cuda_stream
    attrs = []
    if k.get("CL", 0) > 1:
        a = driver.CUlaunchAttribute()
        a.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        a.value.clusterDim.x, a.value.clusterDim.y, a.value.clusterDim.z = k["CL"], 1, 1
        attrs.append(a)
    lc.attrs = attrs
    lc.numAttrs = len(attrs)
    vp = ctypes.c_void_p
    tensors = (data["qkv"], data["positions"], data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"],
               data["k_scale"], data["v_scale"], data["page_table"], rope_table, ws_o, ws_lse)
    values = tuple(t.data_ptr() for t in tensors) + (data["page_table"].stride(0), splits, q_mult, eps,
                                                    (out if out is not None else ws_o).data_ptr(),
                                                    lastm_counters(ws_o.device).data_ptr())
    types = (vp,) * len(tensors) + (ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, vp, vp)
    _infra._cuda_check(driver.cuLaunchKernelEx(lc, func, (values, types), 0))
