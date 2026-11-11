"""Warp-specialized GEMM1, milestone 1: one expert, T tokens, against a float64 reference.

A CTA takes 64 gate + 64 up rows (4 tile pairs of v35's layout) for all T tokens:
  - producer warpgroup (warps 0-3): warp w loads and decodes pair 4 * blockIdx + w as tile_loop
    does, and writes the BF16 tiles to a shared ring with stmatrix (the decode's registers are
    stmatrix's fragments), K-major core matrices: [tile][k step][half][row group][8 rows][16 B];
  - NC consumer warpgroups of NTOK tokens each read the same decoded stage, so all T = NC * NTOK
    tokens share one decode. Each stages its own activations (x in staging order: wg_perm) with
    cp.async into a private ring and runs wgmma m64n(NTOK)k16 with A and B in shared memory;
  - mbarriers full[s] (4 producer warps) and empty[s] (NC consumers) between them; setmaxnreg moves
    registers from the producers to the consumers' accumulators.
Under -DOKC_EMULATE (sm75) mbarriers, stmatrix, cp.async and wgmma have plain versions, so the
dev box checks the indexing; the H100 bench checks the real thing.

    python ws_micro.py local            # emulated, small, on the dev box
    python ws_micro.py submission > ‹tmp›/okc-moe/ws1/submission.py   # bench diagnostic (prepare())
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# v35 (interleaved gate/up tile pairs) is not published; point BASE at a copy of it to run this.
BASE = os.path.join(HERE, "..", "variants", "v35_interleaved_pairs.py")

def load_base(path=BASE, name="v35"):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---8<--- micro (from here on, embedded as is in the bench diagnostic)
import sys


def wgmma_ss_asm(n: int) -> str:
    """The wgmma m64n(n)k16 bf16 SS call (A and B by descriptor) as an inline asm body."""
    nd = n // 2
    regs = ",".join(f"%{i}" for i in range(nd))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nd))
    return (f'asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nd + 2}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nd}, %{nd + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs}\n'
            f'    : "l"(da), "l"(db), "r"(1));')


WS_SRC = r'''
// ---------------------------------------------------------------- warp-specialized GEMM1 (milestone 1)
#ifndef WS_NTOK
#define WS_NTOK 128
#endif
#ifndef WS_NC
#define WS_NC 2
#endif
#ifndef WS_STAGES
#define WS_STAGES 3
#endif
#define WS_A_BYTES (2 * 8192)                       // gate + up, 64 rows x 64 k BF16 each
#define WS_B_BYTES (WS_NTOK * 128)                  // one consumer's tokens x 64 k BF16
#define WS_STAGE_BYTES (WS_A_BYTES + WS_NC * WS_B_BYTES)

#ifndef OKC_EMULATE
__device__ __forceinline__ void ws_mb_init(unsigned long long* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"((u32)__cvta_generic_to_shared(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void ws_mb_arrive(unsigned long long* b) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"((u32)__cvta_generic_to_shared(b)) : "memory");
}
// Bounded: a deadlock traps after ~2^24 tries (an error in seconds, not a 20-minute timeout).
__device__ __forceinline__ void ws_mb_wait(unsigned long long* b, int parity) {
  for (int tries = 0;; ++tries) {
    u32 done;
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"((u32)__cvta_generic_to_shared(b)), "r"(parity) : "memory");
    if (done) return;
    if (tries > (1 << 24)) asm volatile("trap;");
  }
}
__device__ __forceinline__ void ws_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void ws_proxy_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// stmatrix x4: lane i gives the row address of matrix i / 8, row i % 8; register j is matrix j.
__device__ __forceinline__ void ws_stmatrix4(u32 addr_of_my_row, u32 r0, u32 r1, u32 r2, u32 r3, int, u32, u32, u32, u32) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
               :: "r"(addr_of_my_row), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
__device__ __forceinline__ void ws_cp16(u32 smem, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem), "l"(g) : "memory");
}
__device__ __forceinline__ void ws_cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void ws_cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void ws_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ unsigned long long ws_desc(u32 saddr, u32 lbo, u32 sbo) {
  return (unsigned long long)((saddr >> 4) & 0x3FFFu) | ((unsigned long long)((lbo >> 4) & 0x3FFFu) << 16) |
         ((unsigned long long)((sbo >> 4) & 0x3FFFu) << 32);
}
__device__ __forceinline__ void ws_mma(float* d, u32 a_addr, u32 b_addr) {
  const unsigned long long da = ws_desc(a_addr, 1024, 128);
  const unsigned long long db = ws_desc(b_addr, 16 * WS_NTOK, 128);
  @WGMMA_SS@
}
__device__ __forceinline__ void ws_mma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void ws_mma_commit_wait() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}
#define WS_SETMAXNREG_DEC(n) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(n))
#define WS_SETMAXNREG_INC(n) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(n))
#else
// Below sm90: a counter and a completed-phase count in shared memory (as in the attention kernel).
__device__ __forceinline__ void ws_mb_init(unsigned long long* b, int count) {
  unsigned* w = (unsigned*)b;
  w[0] = 0; w[1] = 0; w[2] = count;
}
__device__ __forceinline__ void ws_mb_arrive(unsigned long long* b) {
  unsigned* w = (unsigned*)b;
  __threadfence_block();
  if (atomicAdd(&w[0], 1u) + 1 == w[2]) {
    w[0] = 0;
    __threadfence_block();
    atomicAdd(&w[1], 1u);
  }
}
__device__ __forceinline__ void ws_mb_wait(unsigned long long* b, int parity) {
  volatile unsigned* w = (volatile unsigned*)b;
  while ((w[1] & 1u) == (unsigned)parity) { }
  __threadfence_block();
}
__device__ __forceinline__ void ws_fence_init() {}
__device__ __forceinline__ void ws_proxy_fence() {}
__device__ __forceinline__ void ws_st32(u32 addr, u32 v) { asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(v) : "memory"); }
// Every lane stores its own 4 bytes of each matrix (row g of matrix j at rowaddr(j, g)).
__device__ __forceinline__ void ws_stmatrix4(u32, u32 r0, u32 r1, u32 r2, u32 r3, int c, u32 a0, u32 a1, u32 a2, u32 a3) {
  ws_st32(a0 + 4 * c, r0); ws_st32(a1 + 4 * c, r1); ws_st32(a2 + 4 * c, r2); ws_st32(a3 + 4 * c, r3);
  __syncwarp();
}
__device__ __forceinline__ void ws_cp16(u32 smem, const void* g) {
  const uint4 v = *reinterpret_cast<const uint4*>(g);
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(smem), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void ws_cp_commit() {}
template <int N> __device__ __forceinline__ void ws_cp_wait() {}
__device__ __forceinline__ void ws_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ float ws_ld_bf(u32 addr) {
  unsigned short v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(addr));
  return __uint_as_float((u32)v << 16);
}
__device__ __noinline__ void ws_mma(float* d, u32 a_addr, u32 b_addr) {
  const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3, wi = (threadIdx.x >> 5) & 3;
  for (int j = 0; j < WS_NTOK / 8; ++j)
    for (int e = 0; e < 4; ++e) {
      const int rr = 16 * wi + g + 8 * (e >> 1), n = 8 * j + 2 * c + (e & 1);
      float sum = 0.f;
      for (int i = 0; i < 16; ++i) {
        const float av = ws_ld_bf(a_addr + (i >> 3) * 1024 + (rr >> 3) * 128 + (rr & 7) * 16 + (i & 7) * 2);
        const float bv = ws_ld_bf(b_addr + (i >> 3) * 16 * WS_NTOK + (n >> 3) * 128 + (n & 7) * 16 + (i & 7) * 2);
        sum += av * bv;
      }
      d[4 * j + e] += sum;
    }
}
__device__ __forceinline__ void ws_mma_fence() {}
__device__ __forceinline__ void ws_mma_commit_wait() {}
#define WS_SETMAXNREG_DEC(n)
#define WS_SETMAXNREG_INC(n)
#endif

// One expert: x_perm [T][K] in staging order (wg_perm), w/sf in v35's pair layout, out [T][N] BF16.
extern "C" __global__ void __launch_bounds__(128 * (1 + WS_NC), 1)
ws_gemm1(const u16* __restrict__ xperm, const uint4* __restrict__ w, const uint4* __restrict__ sf,
         const float* __restrict__ gs, u16* __restrict__ out, int K, int N, unsigned long long* prof) {
  extern __shared__ __align__(1024) unsigned char ws_smem[];
  // Barriers 16 bytes apart: the emulated one (below sm90) takes 12.
  unsigned long long* full = reinterpret_cast<unsigned long long*>(ws_smem + WS_STAGES * WS_STAGE_BYTES);
  unsigned long long* empty = full + 2 * WS_STAGES;
  const int t = threadIdx.x, lane = t & 31, g = lane >> 2, c = lane & 3;
  const int KC = K >> 6;
  const u32 sbase = (u32)__cvta_generic_to_shared(ws_smem);
  if (t < WS_STAGES) {
    ws_mb_init(full + 2 * t, 4);       // the producer warps
    ws_mb_init(empty + 2 * t, WS_NC);  // the consumers
  }
  ws_fence_init();
  __syncthreads();
  const long long t_start = clock64();
  long long waited = 0;   // CTA 0, lane 0 of the first warp of each role: cycles spent in mbarrier waits
  if (t < 128) {
    // ------------------------------------------------ producer: decode to BF16 in shared memory
    // Registers: 168 at launch x 384 threads = 104 x 128 (producer) + 200 x 256 (consumers). The
    // consumers' inc only completes once the producer's dec has freed exactly that many.
    WS_SETMAXNREG_DEC(104);
    const int wp = t >> 5;
    const int pair = 4 * blockIdx.x + wp;
    const uint4* wt0 = w + ((long long)pair * KC) * 64 + lane;
    const uint4* st0 = sf + ((long long)pair * (KC >> 3)) * 64 + lane;
    const u32 sbias = @SCALE_FAST_BIAS@u | ((u32)K >> 30);
    for (int grp = 0; grp < (KC >> 3); ++grp) {
      const uint4 s0 = ldg_keep(st0 + grp * 64), s1 = ldg_keep(st0 + grp * 64 + 32);
      uint4 a0[8], a1[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        a0[u] = ldg_stream(wt0 + (grp * 8 + u) * 64);
        a1[u] = ldg_stream(wt0 + (grp * 8 + u) * 64 + 32);
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int ch = grp * 8 + u, s = ch % WS_STAGES;
        if (ch >= WS_STAGES) {
          const long long t0 = clock64();
          ws_mb_wait(empty + 2 * s, ((ch / WS_STAGES) - 1) & 1);
          waited += clock64() - t0;
        }
        const u32 p0g = scale_of<false>(word(s0, u >> 1), 2 * (u & 1), sbias);
        const u32 p0g8 = scale_of<false>(word(s0, u >> 1), 2 * (u & 1) + 1, sbias);
        const u32 p1g = scale_of<false>(word(s1, u >> 1), 2 * (u & 1), sbias);
        const u32 p1g8 = scale_of<false>(word(s1, u >> 1), 2 * (u & 1) + 1, sbias);
        const u32 abase = sbase + s * WS_STAGE_BYTES;
        // matrix j of warp wp's 16 x 16 tile: row group 2 wp + (j & 1), k half j >> 1
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
          for (int tt = 0; tt < 2; ++tt) {
            u32 r0, r1, r2, r3;
            decode(word(tt ? a1[u] : a0[u], kk), tt ? p1g : p0g, tt ? p1g8 : p0g8, r0, r1, r2, r3);
            const u32 tb = abase + tt * 8192 + kk * 2048;
            const int mj = lane >> 3, mr = lane & 7;
            const u32 myrow = tb + (mj >> 1) * 1024 + (2 * wp + (mj & 1)) * 128 + mr * 16;
            ws_stmatrix4(myrow, r0, r1, r2, r3, c,
                         tb + (2 * wp) * 128 + g * 16, tb + (2 * wp + 1) * 128 + g * 16,
                         tb + 1024 + (2 * wp) * 128 + g * 16, tb + 1024 + (2 * wp + 1) * 128 + g * 16);
          }
        }
        ws_proxy_fence();
        __syncwarp();
        if (lane == 0) ws_mb_arrive(full + 2 * s);
      }
    }
  } else {
    // ------------------------------------------------ consumers: wgmma on the decoded stages
    WS_SETMAXNREG_INC(200);
    const int ci = (t - 128) >> 7, ct = (t - 128) & 127, wi = ct >> 5;
    const int tok0 = ci * WS_NTOK;
    const u32 bb = sbase + WS_A_BYTES + ci * WS_B_BYTES;   // + s * WS_STAGE_BYTES
    auto stage_b = [&](int ch) {
      const u32 dst = bb + (ch % WS_STAGES) * WS_STAGE_BYTES;
      for (int idx = ct; idx < WS_NTOK * 8; idx += 128) {
        const int n = idx >> 3, ph = idx & 7;
        ws_cp16(dst + (ph >> 1) * 32 * WS_NTOK + (ph & 1) * 16 * WS_NTOK + (n >> 3) * 128 + (n & 7) * 16,
                xperm + (long long)(tok0 + n) * K + ch * 64 + ph * 8);
      }
    };
    float dg[WS_NTOK / 2], du[WS_NTOK / 2];
#pragma unroll
    for (int i = 0; i < WS_NTOK / 2; ++i) dg[i] = du[i] = 0.f;
#pragma unroll
    for (int i = 0; i < WS_STAGES - 1; ++i) {
      if (i < KC) stage_b(i);
      ws_cp_commit();
    }
    for (int ch = 0; ch < KC; ++ch) {
      const int s = ch % WS_STAGES;
      if (ch + WS_STAGES - 1 < KC) stage_b(ch + WS_STAGES - 1);   // its slot's MMAs finished last step
      ws_cp_commit();
      ws_cp_wait<WS_STAGES - 1>();   // this thread's copies of chunk ch have landed
      ws_proxy_fence();
      ws_bar(1 + ci, 128);           // ... and every thread's of this consumer
      const long long t0 = clock64();
      ws_mb_wait(full + 2 * s, (ch / WS_STAGES) & 1);
      waited += clock64() - t0;
      ws_mma_fence();
      const u32 ab = sbase + s * WS_STAGE_BYTES, bs = bb + s * WS_STAGE_BYTES;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        ws_mma(dg, ab + kk * 2048, bs + kk * 32 * WS_NTOK);
        ws_mma(du, ab + 8192 + kk * 2048, bs + kk * 32 * WS_NTOK);
      }
      ws_mma_commit_wait();
      ws_bar(1 + ci, 128);           // every warp of this consumer is done reading the stage
      if (ct == 0) ws_mb_arrive(empty + 2 * s);
    }
    const float scale = gs[0] * 1.1805916207174113e21f;
#pragma unroll
    for (int j = 0; j < WS_NTOK / 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int n = tok0 + 8 * j + 2 * c + (e & 1);
        const int col = 64 * blockIdx.x + 16 * wi + g + 8 * (e >> 1);
        const float gv = dg[4 * j + e] * scale, uv = du[4 * j + e] * scale;
        out[(long long)n * N + col] = f2bf(gv / (1.f + __expf(-gv)) * uv);
      }
    if (blockIdx.x == 0 && t == 128) {
      prof[1] = (unsigned long long)waited;
      prof[2] = (unsigned long long)(clock64() - t_start);
    }
  }
  if (blockIdx.x == 0 && t == 0) prof[0] = (unsigned long long)waited;
}
'''


def ws_source(m, ntok, nc, stages):
    src = WS_SRC.replace("@WGMMA_SS@", wgmma_ss_asm(ntok)).replace("@SCALE_FAST_BIAS@", f"0x{m.SCALE_FAST_BIAS:08X}")
    # The defines go after CUDA_SRC: torch's wrapper puts 'extern "C"' before a source that starts
    # with a preprocessor line.
    return m.CUDA_SRC + f"\n#define WS_NTOK {ntok}\n#define WS_NC {nc}\n#define WS_STAGES {stages}\n" + src


def wg_perm_cols(k: int):
    """Column order of x in staging order: position p of a 64-chunk holds real k perm[p]."""
    import torch
    perm = torch.empty(k, dtype=torch.long)
    for col in range(k):
        kk_ = col & 63
        p = (col & ~63) | ((((kk_ >> 2) & 3) * 2 + ((kk_ >> 1) & 1)) * 8 + (kk_ >> 4) * 2 + (kk_ & 1))
        perm[p] = col
    return perm


def run(m, dev, I, H, ntok, nc, stages, emulate, log=print):
    import torch
    nvfp4 = sys.modules.get("nvfp4") or __import__("nvfp4")
    T = ntok * nc
    gen = torch.Generator(device="cpu").manual_seed(11)
    wf = (torch.randn((2 * I, H), generator=gen) * 0.05).to(dev)
    codes, sfb, gsv = nvfp4.quantize(wf)
    wdq = nvfp4.dequantize(codes, sfb, gsv).double()                     # [2I, H]
    x = torch.randn((T, H), generator=gen).to(torch.bfloat16).to(dev)
    hh = x.double() @ wdq.T                                              # [T, 2I]
    g_, u_ = hh[:, :I], hh[:, I:]
    ref = (g_ / (1 + torch.exp(-g_)) * u_)
    assert not m._needs_exact(sfb), "the producer decodes scales with the fast path"
    w13 = m._pair_tiles(m._pack_per_expert(codes[None]), gated=True)
    sf13 = m._pair_tiles(m.pack_scales(m._scale_bytes(sfb[None])), gated=True)
    xperm = x[:, wg_perm_cols(H).to(dev)].contiguous()
    out = torch.zeros((T, I), dtype=torch.bfloat16, device=dev)
    opts = ["-std=c++17"] + (["-DOKC_EMULATE"] if emulate else [])
    kw = {} if emulate else {"compute_capability": "90a"}
    k = torch.cuda._compile_kernel(kernel_source=ws_source(m, ntok, nc, stages), kernel_name="ws_gemm1",
                                   nvcc_options=opts, **kw)
    smem = stages * (2 * 8192 + nc * ntok * 128) + 4 * stages * 8
    if smem >= 48 * 1024:
        k.set_shared_memory_config(smem)
    gsd = torch.as_tensor([float(gsv)], dtype=torch.float32, device=dev)
    prof = torch.zeros(4, dtype=torch.int64, device=dev)
    wargs = [xperm, w13.view(torch.int32), sf13, gsd, out, H, I, prof]
    k(grid=(I // 64, 1, 1), block=(128 * (1 + nc), 1, 1), shared_mem=smem, args=wargs)
    torch.cuda.synchronize()
    err = (out.double() - ref).norm() / ref.norm()
    log(f"DIAG ws I={I} H={H} T={T} ntok={ntok} nc={nc} stages={stages}: relative L2 error {float(err):.3e}, "
        f"max |out| {float(out.float().abs().max()):.3g}, out[0,:4] {[round(float(v), 4) for v in out[0, :4]]}, "
        f"ref[0,:4] {[round(float(v), 4) for v in ref[0, :4]]}")
    pr = [int(v) for v in prof.cpu()]
    log(f"DIAG ws waits (CTA 0): producer {pr[0] / max(pr[2], 1):.1%}, consumer 0 {pr[1] / max(pr[2], 1):.1%} "
        f"of {pr[2]} cycles")
    if not emulate:
        def timeit(fn, n=20):
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(3):
                fn()
            ev0.record()
            for _ in range(n):
                fn()
            ev1.record()
            torch.cuda.synchronize()
            return ev0.elapsed_time(ev1) / n * 1000
        t_ws = timeit(lambda: k(grid=(I // 64, 1, 1), block=(128 * (1 + nc), 1, 1), shared_mem=smem, args=wargs))
        # The same expert through v35's moe_gemm1 (blocks of 16, a warp per (block, 16 columns)).
        k1 = torch.cuda._compile_kernel(kernel_source=m.CUDA_SRC, kernel_name="moe_gemm1", nvcc_options=["-std=c++17"])
        nb = (T + 15) // 16
        tok = torch.arange(T, dtype=torch.int32, device=dev)
        bexp = torch.zeros(nb, dtype=torch.int32, device=dev)
        brow = (torch.arange(nb, dtype=torch.int32, device=dev) * 16)
        bcnt = torch.clamp(T - brow, max=16).to(torch.int32)
        nblk = torch.tensor([nb, T], dtype=torch.int32, device=dev)
        out2 = torch.zeros((T, I), dtype=torch.bfloat16, device=dev)
        norm_w = torch.zeros(H, dtype=torch.bfloat16, device=dev)
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        cg = min(sms * 4, (nb * (I // 16) + 3) // 4)
        a1 = [x, tok, bexp, brow, bcnt, nblk, w13.view(torch.int32), sf13, gsd, norm_w, out2, T, 0, H, I, 1]
        k1(grid=(cg, 1, 1), block=(128, 1, 1), args=a1)
        torch.cuda.synchronize()
        err2 = (out2.double() - ref).norm() / ref.norm()
        t_v = timeit(lambda: k1(grid=(cg, 1, 1), block=(128, 1, 1), args=a1))
        log(f"DIAG ws T={T}: ws {t_ws:.1f} us ({I // 64} CTAs) vs v35 moe_gemm1 {t_v:.1f} us ({cg} CTAs, err {float(err2):.2e})")
    return float(err)


# ---8<--- end micro


SUB_TAIL = r'''

# ---------------------------------------------------------------- warp-specialized GEMM1 micro-test (bench only)
_WS_MICRO = @MICRO@


def prepare(weights: Weights):
    state = _prepare(weights)
    import traceback, types
    try:
        mod = types.ModuleType("ws_micro")
        exec(_WS_MICRO, mod.__dict__)
        me = sys.modules[__name__]
        for ntok, nc, stages in ((16, 2, 3), (64, 2, 3), (128, 2, 3)):
            mod.run(me, weights["w13"].device, 4096, 4096, ntok, nc, stages, emulate=False,
                    log=lambda s: print(s, flush=True))
    except Exception as exc:
        print(f"DIAG ws failed: {exc!r}", traceback.format_exc()[-2500:], flush=True)
    return state
'''


def submission():
    src = open(BASE).read()
    assert src.count("\ndef prepare(weights: Weights):") == 1
    src = src.replace("\ndef prepare(weights: Weights):", "\ndef _prepare(weights: Weights):")
    if "\nimport sys\n" not in src:
        src = src.replace("\nimport torch\n", "\nimport sys\nimport torch\n", 1)
    me = open(os.path.abspath(__file__)).read()
    micro = me.split("# ---8<--- micro")[1].split("\n", 1)[1].split("# ---8<--- end micro")[0]
    return src + SUB_TAIL.replace("@MICRO@", repr(micro))


if __name__ == "__main__":
    if sys.argv[1] == "local":
        sys.path.insert(0, os.path.join(HERE, ".."))
        import torch
        m = load_base()
        m.CUDA_OPTIONS = ["-std=c++17", "-DOKC_EMULATE"]
        run(m, torch.device("cuda"), 256, 512, 16, 2, 2, emulate=True)
    else:
        print(submission())
