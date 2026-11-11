# muse_moe_w4a16: roofline per shape (H200)

Data in [`evidence/roofline-moe.json`](../../evidence/roofline-moe.json). The floor is the one in
TASK.md: move the minimum bytes at 4.8 TB/s or do the arithmetic at 989 TFLOP/s, whichever takes
longer. Bandwidth dominates in all six shapes. "% peak" is floor / time: the fraction of the
4.8 TB/s that is used.

| load | floor µs | SGLang µs | % peak SGLang | mine µs (0.578x) | % peak mine | µs for top 50 | % peak for top 50 |
|---|---|---|---|---|---|---|---|
| decode 1 | 11.4 | 78.8 | 15% | 141.9 | 8% | 37.0 | 31% |
| decode 16 | 52.5 | 184.9 | 28% | 357.5 | 15% | 86.9 | 60% |
| decode 32 | 70.8 | 203.6 | 35% | 447.0 | 16% | 95.7 | 74% |
| verify 1 | 37.3 | 147.6 | 25% | 238.5 | 16% | 69.4 | 54% |
| verify 16 | 101.3 | 459.7 | 22% | 705.8 | 14% | 216.1 | 47% |
| verify 32 | 108.9 | 614.4 | 18% | 966.4 | 11% | 288.8 | 38% |

Geometric mean of % peak: SGLang 23%, mine 13%, top 50 49%, top 10 54%, leader 58%. The
theoretical ceiling is 4.4x.

**Reading.**
- SGLang leaves a lot of room: it uses less than a third of the bandwidth. That is why the leader
  reaches 2.5x.
- The top 50 needs the NVFP4 weights read at ~50% of peak on average, and at ~75% at decode 32.
  That requires a kernel that keeps all 132 SMs reading weights at all times: with one token and
  8 experts, the grid has to split N and K, not only distribute experts.
- I am at 13%, below the 50% set by the plan's rule. If `ncu` finds no fixable cause within
  2 h (short grid, spills, unvectorized loads), the work moves to a Marlin-style W4A16 GEMM in
  CUDA.
- At decode 1 the floor (11 µs) is close to the cost of launching 3 or 4 kernels. There, fusing
  everything into one persistent kernel matters: routing, GEMM1+SwiGLU, GEMM2, norm and sum.
