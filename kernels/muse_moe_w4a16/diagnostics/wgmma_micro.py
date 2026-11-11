"""Bench micro-test for the wgmma GEMM (milestone 0): submission.py (v24) with a self-test at
the end of prepare() that runs, on the target GPU (sm_90a), one CTA doing four
wgmma.mma_async m64n32k16 steps (one 64-k chunk):
  - A (64 x 16 per step) from registers, in the per-warp mma.sync m16n8k16 A fragment layout
    the decode produces (warp w holds rows 16w..16w+15);
  - B (16 x 32 per step) from shared memory, K-major without swizzle (core matrices of 8 tokens
    x 16 bytes; LBO 512 B between the two k halves, SBO 128 B between token groups), staged from
    activation rows with the packing's k permutation (word w of a chunk -> step (w >> 1) & 3,
    half w & 1, position w >> 3);
  - D read back as rows 16w + g (+8), columns 8j + 2c (+1).
and prints DIAG lines with the error against a float64 reference. The judge's tests and
timings run as usual on v24; only the log matters.

    python wgmma_micro.py > ‹tmp›/okc-moe/wgmma0/submission.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

MICRO = r'''

# ---------------------------------------------------------------- wgmma micro-test (bench only)
_WG_SRC = r"""
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u64 desc_b(const void* smem, u32 lbo, u32 sbo) {
  const u32 a = (u32)__cvta_generic_to_shared(smem);
  return (u64)((a >> 4) & 0x3FFFu) | ((u64)((lbo >> 4) & 0x3FFFu) << 16) |
         ((u64)((sbo >> 4) & 0x3FFFu) << 32);  // base offset 0, layout 0 (no swizzle)
}

// A: [4 steps][64 rows][16 k] bf16 bits (u16), row-major per step. X: [32 tokens][64 k] bf16.
// D: [64 rows][32 tokens] f32.
extern "C" __global__ void __launch_bounds__(128) wg_test(const unsigned short* A, const u32* X, float* D) {
  __shared__ __align__(128) u32 sb[4 * 1024 / 4];  // 4 steps x 1 KB
  const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
  // Stage B: token n, word wd (pair of k 2wd, 2wd + 1) of the chunk -> step kk, half h, pos c.
  for (int i = t; i < 32 * 32; i += 128) {
    const int n = i >> 5, wd = i & 31;
    const int kk = (wd >> 1) & 3, h = wd & 1, pos = wd >> 3;
    const int ng = n >> 3, r = n & 7;
    sb[((((kk * 2 + h) * 4 + ng) * 128) + r * 16 + pos * 4) / 4] = X[n * 32 + wd];
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  float d[16];
  for (int j = 0; j < 16; ++j) d[j] = 0.f;
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  for (int kk = 0; kk < 4; ++kk) {
    const unsigned short* a = A + kk * 64 * 16;
    const int r0 = 16 * w + g, r1 = r0 + 8;
    const u32 a0 = a[r0 * 16 + 2 * c] | ((u32)a[r0 * 16 + 2 * c + 1] << 16);
    const u32 a1 = a[r1 * 16 + 2 * c] | ((u32)a[r1 * 16 + 2 * c + 1] << 16);
    const u32 a2 = a[r0 * 16 + 2 * c + 8] | ((u32)a[r0 * 16 + 2 * c + 9] << 16);
    const u32 a3 = a[r1 * 16 + 2 * c + 8] | ((u32)a[r1 * 16 + 2 * c + 9] << 16);
    const u64 db = desc_b(reinterpret_cast<const char*>(sb) + kk * 1024, 512, 128);
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(1));
  }
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  for (int j = 0; j < 4; ++j) {
    const int col = 8 * j + 2 * c, r0 = 16 * w + g;
    D[r0 * 32 + col] = d[4 * j];
    D[r0 * 32 + col + 1] = d[4 * j + 1];
    D[(r0 + 8) * 32 + col] = d[4 * j + 2];
    D[(r0 + 8) * 32 + col + 1] = d[4 * j + 3];
  }
}
"""


def _wgmma_micro(dev):
    import traceback
    try:
        print("DIAG gpu", torch.cuda.get_device_name(dev), "cc", torch.cuda.get_device_capability(dev), flush=True)
        k = torch.cuda._compile_kernel(kernel_source=_WG_SRC, kernel_name="wg_test",
                                       compute_capability="90a", nvcc_options=["-std=c++17"])
        gen = torch.Generator(device="cpu").manual_seed(7)
        a = torch.randn((4, 64, 16), generator=gen).to(torch.bfloat16)
        x = torch.randn((32, 64), generator=gen).to(torch.bfloat16)
        # Reference: step kk, k position i <-> real k 16 * ((i % 8) >> 1) + 4 kk + 2 (i >> 3) + (i & 1)
        ref = torch.zeros((64, 32), dtype=torch.float64)
        for kk in range(4):
            for i in range(16):
                kr = 16 * ((i % 8) >> 1) + 4 * kk + 2 * (i >> 3) + (i & 1)
                ref += a[kk, :, i].double()[:, None] * x[:, kr].double()[None, :]
        A = a.view(torch.int16).to(dev).contiguous()
        X = x.contiguous().view(torch.int32).to(dev)
        D = torch.zeros((64, 32), dtype=torch.float32, device=dev)
        k(grid=(1, 1, 1), block=(128, 1, 1), args=[A, X, D])
        torch.cuda.synchronize()
        got = D.double().cpu()
        err = (got - ref).abs()
        print("DIAG wgmma max |err|", float(err.max()), "max |ref|", float(ref.abs().max()),
              "bad (>1e-2)", int((err > 1e-2).sum()), "of", err.numel(), flush=True)
        print("DIAG wgmma got[0,:8]", [round(float(v), 3) for v in got[0, :8]], flush=True)
        print("DIAG wgmma ref[0,:8]", [round(float(v), 3) for v in ref[0, :8]], flush=True)
        print("DIAG wgmma got[9,:8]", [round(float(v), 3) for v in got[9, :8]], flush=True)
        print("DIAG wgmma ref[9,:8]", [round(float(v), 3) for v in ref[9, :8]], flush=True)
    except Exception as exc:
        print(f"DIAG wgmma failed: {exc!r}", traceback.format_exc()[-2000:], flush=True)


def prepare(weights: Weights):
    state = _prepare(weights)
    _wgmma_micro(weights["w13"].device)
    return state
'''


def main():
    src = open(os.path.join(HERE, "..", "submission.py")).read()
    assert src.count("\ndef prepare(weights: Weights):") == 1
    src = src.replace("\ndef prepare(weights: Weights):", "\ndef _prepare(weights: Weights):")
    print(src + MICRO)


if __name__ == "__main__":
    main()
