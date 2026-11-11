// KERNEL_A_WS_CUDA_SOURCE: copied from kernel.py, lines 994-1384.
// The challenge takes a single Python file, so this is a copy for reading with syntax
// highlighting; kernel.py is the source. It is compiled in-process with NVRTC, and
// kernel.py passes the -D configuration constants for each shape.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// verify_ws: split-KV attention for verify tiles (8 tokens x 8 query heads = 64 rows per KV head),
// warp-specialized, sm_90a. -D from _cufa_consts: HQ HKV L STAGES PAGE WINDOW ROPE WS=1 KVP=1
// PNOWAIT=1 PSMEM=1. One producer warp gathers each block's scattered K and V rows with cp.async
// into a STAGES-deep ring and signals full[s] with cp.async.mbarrier.arrive.noinc (it never waits on
// its own copies); one consumer warpgroup per KV head runs S = Q K^T as wgmma m64n64k32 on E4M3
// from shared memory, converts V to FP16 while the wgmma runs, and does P V with mma.sync m16n8k16.
// Each CTA writes its split's normalized partial and log-sum-exp for the combine kernel.
#define D 64
#define BN 64                          // keys per block
#define NS (BN / 2)                    // scores per thread (the m64nBN accumulator)
#define ROWS 64                        // query rows per KV head: one warpgroup m64
#define NWG KVP                        // consumer warpgroups, one per KV head
#define NCONS (128 * NWG)              // consumer threads
#define NT (NCONS + 128)               // + the producer warpgroup (warp NCONS / 32 copies)
#define G (HQ / HKV)
#define QKV_STRIDE ((2 * HQ + 2 * HKV) * D)
#define ROW (HKV * D)
#define KROW (KVP * D)
#define TILE (BN * D)                  // bytes of an E4M3 key/value tile (BN x 64)
#define QTILE (ROWS * D)               // bytes of an E4M3 query tile (64 x 64)
#define VP (D + 8)                     // FP16 V rows (halves), padded: ldmatrix without bank conflicts
#define NEG_INF __int_as_float(0xff800000)
// Registers: 64 at launch x 256 threads (4 CTAs/SM), then setmaxnreg moves them to 24 for the
// producer warpgroup and 104 for the consumer warpgroup (24 x 128 + 104 x 128 = 64 x 256).
#define MINB 4                         // CTAs per SM (__launch_bounds__)
#define CREGS 104                      // consumer registers (setmaxnreg.inc)
#define PREGS 24                       // producer registers (setmaxnreg.dec)
#define VHBUF 2                        // FP16 V buffers per consumer warpgroup
// The page entries go to a shared-memory ring by cp.async, PQS blocks ahead, and each block waits
// for its own with wait_group. A register queue made the compiler move every loaded entry into its
// slot right away, so each load was waited for where it was issued.
#define PQS 4                          // blocks of page entries in flight
#define PSL (PQS + 1)                  // ring slots

#define OFF_Q 0
#define OFF_K (OFF_Q + NWG * QTILE)    // a Q tile per consumer warpgroup
#define OFF_V (OFF_K + STAGES * KVP * TILE)
#define OFF_VH (OFF_V + STAGES * KVP * TILE)
#define OFF_B (OFF_VH + NWG * VHBUF * BN * VP * 2)   // mbarriers: full[STAGES], empty[STAGES]
#define OFF_PG (OFF_B + (2 * STAGES + 4) * 16)      // 16 bytes per barrier (+ 4 unused slots)
// [PSL][BN] page entries from OFF_PG on; the host's _cufa_smem has the same total.

static_assert(L * G <= ROWS, "one warpgroup holds a head's rows");
static_assert(KVP == 1 && PNOWAIT == 1 && PSMEM == 1, "the only compiled variant (MINB, CREGS, PREGS assume it)");
static_assert(STAGES >= 2, "one block in flight at least");
#define CPR (KROW / 16)                // the producer's lanes per key row (16 bytes each)

__device__ __forceinline__ int cm(int row, int byte) {   // no-swizzle core matrices of a 64 x 64-byte tile
    return (row >> 3) * 512 + (byte >> 4) * 128 + (row & 7) * 16 + (byte & 15);
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void mma_k16(float* d, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *(unsigned*)&h;
}
__device__ __forceinline__ unsigned e4m3x2_to_h2(unsigned short v) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
    return *(unsigned*)&h;
}
// 16-byte copy, zero-filled when !ok (src-size 0).
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_addr(dst)), "l"(src), "r"(ok ? 16 : 0));
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
__device__ __forceinline__ void fence_async_smem() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
// Named barrier for one warpgroup (ids 1 .. KVP; 0 is __syncthreads).
__device__ __forceinline__ void wg_bar(int id) { asm volatile("bar.sync %0, 128;\n" :: "r"(id) : "memory"); }

// mbarriers: arrive count fixed at init, parity waits.
__device__ __forceinline__ void mb_init(unsigned long long* b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_addr(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void mb_arrive(unsigned long long* b) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" :: "r"(smem_addr(b)) : "memory");
}
__device__ __forceinline__ void mb_wait(unsigned long long* b, int parity) {
    asm volatile("{\n.reg .pred done;\nWAIT_%=:\n"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1;\n"
                 "@!done bra WAIT_%=;\n}\n" :: "r"(smem_addr(b)), "r"(parity) : "memory");
}
// Arrive on b once all of this thread's earlier cp.async copies have landed (counted in init).
__device__ __forceinline__ void mb_arrive_cp(unsigned long long* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(smem_addr(b)) : "memory");
}
__device__ __forceinline__ unsigned long long wg_desc(const void* p) {   // no swizzle, LBO 128, SBO 512
    return (unsigned long long)((smem_addr(p) & 0x3FFFF) >> 4) | (8ull << 16) | (32ull << 32);
}
__device__ __forceinline__ void wg_qk(float* d, unsigned long long da, unsigned long long db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {"
                 "%0, %1, %2, %3, %4, %5, %6, %7,"
                 "%8, %9, %10, %11, %12, %13, %14, %15,"
                 "%16, %17, %18, %19, %20, %21, %22, %23,"
                 "%24, %25, %26, %27, %28, %29, %30, %31"
                 "}, %32, %33, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                   "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                   "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                   "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                   "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "l"(da), "l"(db), "r"(acc));
}
__device__ __forceinline__ float2 norm_rope(float2 x, float2 cs, float eps) {
    float inv = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / D + eps);
    x.x *= inv; x.y *= inv;
    return make_float2(x.x * cs.x - x.y * cs.y, x.y * cs.x + x.x * cs.y);
}

extern "C" __global__ void __launch_bounds__(NT, MINB) verify_ws(
    const __nv_bfloat16* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const int* __restrict__ seq_lens, __nv_fp8_e4m3* k_cache, __nv_fp8_e4m3* v_cache,
    const float* __restrict__ k_scale_ptr, const float* __restrict__ v_scale_ptr,
    const int* __restrict__ page_table, const float* __restrict__ rope,
    float* __restrict__ ws_o, float* __restrict__ ws_lse, int pt_stride, int num_splits, float q_mult, float eps)
{
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* Qs = smem + OFF_Q;   // [NWG][64 rows x 64 B] E4M3, core-matrix layout
    unsigned char* Ks = smem + OFF_K;   // [STAGES][KVP][64 keys x 64 B] E4M3, core-matrix layout
    unsigned char* Vr = smem + OFF_V;   // [STAGES][64 keys][KROW] E4M3, row-major
    __half* Vh = (__half*)(smem + OFF_VH);   // [NWG][VHBUF][64 keys][VP] FP16
    // Barrier k at 16-byte steps.
    unsigned long long* full = (unsigned long long*)(smem + OFF_B);       // full[2 k]
    unsigned long long* empty = full + 2 * STAGES;                        // empty[2 k]

    const int split = blockIdx.x, kv0 = blockIdx.y * KVP, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, grp = lane >> 2, tig = lane & 3;
    const float k_scale = *k_scale_ptr, v_scale = *v_scale_ptr;
    const int seq = seq_lens[b], hi = seq + L;
    const int lo = WINDOW > 0 ? max(seq - WINDOW + 1, 0) : 0;
    const int n_blocks = (hi - lo + BN - 1) / BN;
    const int per = (n_blocks + num_splits - 1) / num_splits;
    const int last = n_blocks - split * per, first = max(last - per, 0);
    const int nb = max(last - first, 0);

    // The split holding block 0 writes the step's new keys and values to the cache before anyone
    // copies a block (the barrier below).
    if (first == 0 && last > 0) {
        for (int w = warp; w < KVP * L; w += NT / 32) {
            int j = w % L, kvh = kv0 + w / L, tok = b * L + j;
            long pos = positions[tok], loc = cache_loc[tok];
            const __nv_bfloat162* row = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE);
            float2 kx = __bfloat1622float2(row[(HQ * D + kvh * D) / 2 + lane]);
            float2 vx = __bfloat1622float2(row[((HQ + HKV) * D + kvh * D) / 2 + lane]);
            float2 cs = ROPE ? make_float2(rope[pos * D + lane], rope[pos * D + D / 2 + lane])
                             : make_float2(1.f, 0.f);
            kx = norm_rope(kx, cs, eps);
            *(__nv_fp8x2_storage_t*)(k_cache + loc * ROW + kvh * D + 2 * lane) =
                __nv_cvt_float2_to_fp8x2(make_float2(kx.x / k_scale, kx.y / k_scale), __NV_SATFINITE, __NV_E4M3);
            *(__nv_fp8x2_storage_t*)(v_cache + loc * ROW + kvh * D + 2 * lane) =
                __nv_cvt_float2_to_fp8x2(make_float2(vx.x / v_scale, vx.y / v_scale), __NV_SATFINITE, __NV_E4M3);
        }
        __threadfence();
    }
    if (tid < STAGES) {
        mb_init(full + 2 * tid, 32);         // the producer warp's lanes
        mb_init(empty + 2 * tid, NCONS / 32);   // a lane per consumer warp reading the stage
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    __syncthreads();   // new keys and values in the cache, barriers initialized: the last CTA-wide barrier

    if (warp >= NCONS / 32) {
        // ---------------- producer warpgroup: its first warp copies, the others leave ----------------
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(PREGS));
        if (warp != NCONS / 32) return;
        // Lane l loads the page entries of keys l and l + 32 of each block, PQS blocks ahead into the
        // shared-memory ring (loading them one block ahead left the producer waiting a DRAM latency
        // per block); the copies are spread so that each warp instruction takes whole rows (below).
        int* Pg = (int*)(smem + OFF_PG);
        auto issue_pages = [&](int j) {   // block j's entries into slot j % PSL, one cp.async group
            if (j < nb)
#pragma unroll
                for (int e = 0; e < BN / 32; ++e) {
                    const int key = lane + 32 * e, t = hi - (first + j + 1) * BN + key;
                    if (t >= lo) cp4(Pg + (j % PSL) * BN + key, page_table + (long)b * pt_stride + t / PAGE);
                }
            cp_commit();
        };
#pragma unroll
        for (int u = 0; u < PQS; ++u) issue_pages(u);
        for (int j = 0; j < nb; ++j) {
            const int s = j % STAGES;
            if (j >= STAGES)   // the consumers left j - STAGES
                mb_wait(empty + 2 * s, ((j / STAGES) - 1) & 1);
            cp_wait<PQS - 1>();   // block j's entries (and the copies of blocks before j - PQS + 1)
            __syncwarp();         // ... landed for every lane
#pragma unroll
            for (int i = 0; i < BN * CPR / 32; ++i) {
                const int key = (32 / CPR) * i + lane / CPR, c = lane % CPR, t = hi - (first + j + 1) * BN + key;
                const bool ok = t >= lo;
                const long at = (long)(ok ? Pg[(j % PSL) * BN + key] * PAGE + t % PAGE : 0) * ROW + kv0 * D + c * 16;
                cp16z(Ks + (s * KVP + c / 4) * TILE + cm(key, (c % 4) * 16), k_cache + at, ok);
                cp16z(Vr + s * KVP * TILE + key * KROW + c * 16, v_cache + at, ok);
            }
            mb_arrive_cp(full + 2 * s);   // full[s] completes when every lane's copies so far have landed
            __syncwarp();                 // every lane has read slot j % PSL before it is refilled
            issue_pages(j + PQS);
        }
        cp_wait<0>();   // no thread leaves with cp.async in flight
        return;
    }

    // ---------------- consumer warpgroup wg: KV head kv0 + wg ----------------
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(CREGS));
    const int wg = warp >> 2, h = wg, wl = warp & 3, ct = tid & 127;
    unsigned char* Q = Qs + wg * QTILE;
    // Queries: RMSNorm, RoPE, E4M3 at unit RMS (16 rows per warp; rows past L G are zero).
    float2 qx[ROWS / 4], qcs[ROWS / 4];
    long qpos[ROWS / 4];
#pragma unroll
    for (int i = 0; i < ROWS / 4; ++i) {
        int r = wl + i * 4;
        qx[i] = make_float2(0.f, 0.f);
        qpos[i] = 0;
        if (r < L * G) {
            int tok = b * L + r / G;
            const __nv_bfloat162* src =
                (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE + ((kv0 + h) * G + r % G) * D);
            qx[i] = __bfloat1622float2(src[lane]);
            qpos[i] = positions[tok];
        }
    }
#pragma unroll
    for (int i = 0; i < ROWS / 4; ++i)
        qcs[i] = ROPE ? make_float2(rope[qpos[i] * D + lane], rope[qpos[i] * D + D / 2 + lane]) : make_float2(1.f, 0.f);
#pragma unroll
    for (int i = 0; i < ROWS / 4; ++i) {
        int r = wl + i * 4;
        float2 x = norm_rope(qx[i], qcs[i], eps);
        if (r >= L * G) x = make_float2(0.f, 0.f);
        *(__nv_fp8x2_storage_t*)(Q + cm(r, 2 * lane)) = __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
    }
    fence_async_smem();
    wg_bar(1 + wg);

    const float qsc = q_mult * k_scale;
    int ptok[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) ptok[e] = seq + min((wl * 16 + grp + e * 8) / G, L - 1);
    float o[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m_run[2] = {NEG_INF, NEG_INF}, l_run[2] = {0.f, 0.f};
    unsigned pa[BN / 16][4];   // A fragments of P for k-tiles of 16 keys

    for (int j = 0; j < nb; ++j) {
        const int s = j % STAGES;
        const unsigned char* K = Ks + (s * KVP + h) * TILE;
        const unsigned char* V = Vr + s * KVP * TILE;
        __half* vh = Vh + (wg * VHBUF + j % VHBUF) * BN * VP;
        const int kb0 = hi - (first + j + 1) * BN;
        mb_wait(full + 2 * s, (j / STAGES) & 1);   // block j landed
        fence_async_smem();   // the producer's generic-proxy copies, before wgmma reads K
        // S = Q K^T: accumulator element nt 4 + c is row wl 16 + grp + 8 (c >> 1), key nt 8 + 2 tig + (c & 1).
        float sc[NS];
        // Pinned zeros; the first QK^T wgmma has scale-d 0 anyway.
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            sc[i] = 0.f;
            asm volatile("" : "+f"(sc[i]) :: "memory");
        }
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        wg_qk(sc, wg_desc(Q), wg_desc(K), 0);
        wg_qk(sc, wg_desc(Q + 256), wg_desc(K + 256), 1);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        // Meanwhile this head's V to FP16 (thread ct: 32 bytes of key ct / 2).
#pragma unroll
        for (int vk = 0; vk < BN / 64; ++vk) {   // 32 bytes of key vk 64 + ct / 2 per pass
            const int key = vk * 64 + (ct >> 1), c0 = (ct & 1) * 32;
            const uint4 a = *(const uint4*)(V + key * KROW + h * D + c0);
            const uint4 c = *(const uint4*)(V + key * KROW + h * D + c0 + 16);
            const unsigned w[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
            unsigned hv[16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                hv[2 * i] = e4m3x2_to_h2((unsigned short)w[i]);
                hv[2 * i + 1] = e4m3x2_to_h2((unsigned short)(w[i] >> 16));
            }
            uint4* dst = (uint4*)(vh + key * VP + c0);
#pragma unroll
            for (int i = 0; i < 4; ++i) dst[i] = make_uint4(hv[4 * i], hv[4 * i + 1], hv[4 * i + 2], hv[4 * i + 3]);
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");   // QK^T of j
#pragma unroll
        for (int i = 0; i < NS; ++i) asm volatile("" : "+f"(sc[i]) :: "memory");
        __syncwarp();
        if (lane == 0) mb_arrive(empty + 2 * s);   // this warp is done with stage s (K by wgmma, V converted)
        wg_bar(1 + wg);                        // vh complete for the warpgroup

        const bool edge = kb0 < lo || kb0 + BN - 1 > seq || (WINDOW > 0 && kb0 <= seq + L - 1 - WINDOW);
        float mx[2] = {NEG_INF, NEG_INF};
        if (edge) {
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                int e = (i & 3) >> 1, t = kb0 + (i >> 2) * 8 + 2 * tig + (i & 1), p = ptok[e];
                bool ok = t >= lo && t <= p && (WINDOW == 0 || t > p - WINDOW);
                sc[i] = ok ? sc[i] * qsc : NEG_INF;
                mx[e] = fmaxf(mx[e], sc[i]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                sc[i] *= qsc;
                mx[(i & 3) >> 1] = fmaxf(mx[(i & 3) >> 1], sc[i]);
            }
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
        // sc * 1 - safe as an FMA: the scored build has this form, and sc - safe compiles to other PTX.
#pragma unroll
        for (int nt = 0; nt < BN / 8; ++nt) {
            float p0 = exp2f(fmaf(sc[nt * 4], 1.f, -safe[0])), p1 = exp2f(fmaf(sc[nt * 4 + 1], 1.f, -safe[0]));
            float p2 = exp2f(fmaf(sc[nt * 4 + 2], 1.f, -safe[1])), p3 = exp2f(fmaf(sc[nt * 4 + 3], 1.f, -safe[1]));
            rs[0] += p0 + p1; rs[1] += p2 + p3;
            pa[nt >> 1][(nt & 1) * 2] = pack_h2(p0, p1);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_h2(p2, p3);
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 1);
            rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 2);
            l_run[e] = l_run[e] * alpha[e] + rs[e];
        }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            o[n][0] *= alpha[0]; o[n][1] *= alpha[0]; o[n][2] *= alpha[1]; o[n][3] *= alpha[1];
        }
        // O += P V; vh is rewritten VHBUF blocks on, after a later wg_bar.
#pragma unroll
        for (int kk = 0; kk < BN / 16; ++kk)
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                const int m = lane >> 3;
                unsigned b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, vh + (kk * 16 + (m & 1) * 8 + (lane & 7)) * VP + np * 16 + (m >> 1) * 8);
                mma_k16(o[2 * np], pa[kk], b0, b1);
                mma_k16(o[2 * np + 1], pa[kk], b2, b3);
            }
    }

    // Partials: normalized o and base-2 lse per row, straight to ws (one warp per 16 rows).
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int r = wl * 16 + grp + e * 8;
        if (r >= L * G) continue;
        const long slot = ((long)(b * L + r / G) * HQ + (kv0 + h) * G + r % G) * num_splits + split;
        const float l = l_run[e], inv = l > 0.f ? v_scale / l : 0.f;
#pragma unroll
        for (int n = 0; n < 8; ++n)
            *(float2*)(ws_o + slot * D + n * 8 + 2 * tig) = make_float2(o[n][2 * e] * inv, o[n][2 * e + 1] * inv);
        if (tig == 0) ws_lse[slot] = l > 0.f ? m_run[e] + __log2f(l) : NEG_INF;
    }
}
