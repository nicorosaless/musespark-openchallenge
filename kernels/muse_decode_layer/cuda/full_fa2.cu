// KERNEL_A_CUDA_SOURCE: copied from kernel.py, lines 546-1044.
// The challenge takes a single Python file, so this is a copy for reading with syntax
// highlighting; kernel.py is the source. It is compiled in-process with NVRTC, and
// kernel.py passes the -D configuration constants for each shape.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>

// -D: HQ HKV L KVP WARP_M WARP_K STAGES PAGE WINDOW ROPE MINB (CTAs per SM the registers must fit),
// and in cluster mode also CL NEWSMEM=1 EARLYQ=1 (_cufa_consts).
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
#ifndef CL
#define CL 0
#endif
// NEWSMEM 1: the split holding block 0 also writes the step's new keys and values (E4M3) straight
// into its stage-0 tile and skips their copies, so it neither fences nor reads them back.
#ifndef NEWSMEM
#define NEWSMEM 0
#endif
// EARLYQ 1: the queries' loads go first (they don't depend on seq_lens), then the page entries,
// and the block-0 split writes the step's keys and values last, overlapped with the rest.
#ifndef EARLYQ
#define EARLYQ 0
#endif
static_assert(NEWSMEM == EARLYQ, "the layer sets NEWSMEM and EARLYQ together (cluster mode) or neither");

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
// D += A B over k = 32, A and B in E4M3 (sm89+).
__device__ __forceinline__ void mma_k32_e4m3(float* d, const unsigned* a, const unsigned* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// cp.async: 16 and 4 bytes, global to shared. cp.async goes through the generic proxy, so the
// __syncthreads after cp.async.wait_group is enough; a TMA (async proxy) version would also need
// fence.proxy.async before reusing a stage.
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
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
    __nv_bfloat16* __restrict__ out)
{
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* Qs = smem + OFF_Q;   // [ROWS][QP] E4M3, dims permuted (below)
    unsigned char* Kr = smem + OFF_K;   // [STAGES][BN][RP] E4M3
    unsigned char* Vr = smem + OFF_V;
    int* Pg = (int*)(smem + OFF_P);     // [PRING][BN] page-table entries

    const int split = blockIdx.x, kv0 = blockIdx.y * KVP, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, grp = lane >> 2, tig = lane & 3;
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
            qx[i] = __bfloat1622float2(src[lane]);
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
            kxs[i] = __bfloat1622float2(row[(HQ * D + kvh * D) / 2 + lane]);
            vxs[i] = __bfloat1622float2(row[((HQ + HKV) * D + kvh * D) / 2 + lane]);
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

#if CL
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
            gate_r[i] = make_float2(__bfloat162float(gp[lane]), __bfloat162float(gp[lane + 32]));
        }
    }
#endif
#if !EARLYQ
    // The split holding block 0 writes the step's new keys and values to the cache first; its own
    // copies of block 0 then read them back (after the fence and barrier below).
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
        }
        __threadfence();
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
            for (int i = tid; i < BN * CPK; i += NT) {
                int key = i / CPK, c16 = (i % CPK) * 16, t = hi - (first + j + 1) * BN + key;
                unsigned char* dk = Kr + ((j % STAGES) * BN + key) * RP + c16;
                unsigned char* dv = Vr + ((j % STAGES) * BN + key) * RP + c16;
                if (t < lo) {  // clipped rows: zeros (stale bytes could encode NaN, and 0 * NaN = NaN)
                    *(uint4*)dk = make_uint4(0, 0, 0, 0);
                    *(uint4*)dv = make_uint4(0, 0, 0, 0);
                    continue;
                }
                if (NEWSMEM && first + j == 0 && t >= seq) continue;   // the step's own rows: written above
                long row = (long)Pg[(j % PRING) * BN + key] * PAGE + t % PAGE;
                long at = row * ROW + kv0 * D + c16;
                cp16(dk, k_cache + at);
                cp16(dv, v_cache + at);
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
    cp_wait<0>();
    __syncthreads();   // queries in shared memory, the first page entries landed
    for (int j = 0; j < STAGES - 1; ++j) { issue_kv(j); cp_commit(); }

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
    cp_wait<0>();   // main loop done

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
#if CL
    cooperative_groups::this_cluster().barrier_wait();   // every rank has started (arrived at the top)
#endif
    for (int idx = tid; idx < ROWS * D; idx += NT) {
        int r = idx / D, d = idx % D, mr = r / 16, rl = r % 16;
        int h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        if (rr >= L * G) continue;
        float mm = NEG_INF;
        for (int k = 0; k < WARP_K; ++k) mm = fmaxf(mm, Wm[((k * WARP_M + mr) * 16 + rl) * 2]);
        float safe = mm == NEG_INF ? 0.f : mm, den = 0.f, acc = 0.f;
        for (int k = 0; k < WARP_K; ++k) {
            int w = k * WARP_M + mr;
            float wgt = exp2f(Wm[(w * 16 + rl) * 2] - safe);
            den += wgt * Wm[(w * 16 + rl) * 2 + 1];
            acc += wgt * Wo[(w * 16 + rl) * D + d];
        }
#if CL
        // Row r's partial straight into its owner's buffer (rank rho % CL, slot rho / CL).
        {
            const int rho = h * (L * G) + rr;
            float* mg = cooperative_groups::this_cluster().map_shared_rank((float*)(smem + OFF_MG), rho % CL);
            float* slot = mg + ((rho / CL) * CL + split) * (D + 1);
            slot[d] = den > 0.f ? acc / den * v_scale : 0.f;
            if (d == 0) slot[D] = den > 0.f ? mm + __log2f(den) : NEG_INF;
        }
#else
        int j = rr / G, g = rr % G, head = (kv0 + h) * G + g;
        long slot = ((long)(b * L + j) * HQ + head) * num_splits + split;
        ws_o[slot * D + d] = den > 0.f ? acc / den * v_scale : 0.f;
        if (d == 0) ws_lse[slot] = den > 0.f ? mm + __log2f(den) : NEG_INF;
#endif
    }
#if CL
    // Merge: real row rho on rank rho % CL, warp (rho / CL) % NWARPS; lane holds dims lane, lane + 32.
    const int rank = split;   // the cluster spans the grid's x
    cooperative_groups::this_cluster().sync();   // every rank's partials are in their owners' buffers
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
#endif
}
