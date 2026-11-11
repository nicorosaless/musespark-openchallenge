# bf16_proj: what didn't win on the H200

All numbers are from the judge's H200, in µs per call, and come from labs that run inside
`okc check`: CUDA graph and cold L2, with the fixed cost of an empty graph subtracted. Each table
names the check it comes from. The judge output of most of them is in
[`evidence/runs/muse_bf16_proj/<check>.json`](../../evidence/runs/muse_bf16_proj/). "kf" is each
lab's reference `kernel_fn` (v10 or v11/v11b),
measured in the same run.

The lesson that keeps repeating: the bench H100 (~1.3 TB/s) is memory-bound and almost everything
ties there. On the H200 (~4.8 TB/s), a kernel that trades bytes for instructions or for latency can
lose. What won on the bench was decided on the H200.

## 1. c1 with tensor cores (Triton, `tl.dot`) for m from 8 to 32

Decompresses c1 to BF16 and multiplies with `tl.dot`. Check `ee1a4aa0` (stream-K with MMA, the
best variant):

| shape | v10 (kf) | best c1 MMA |
|---|---|---|
| qkv_gate m8 / m16 / m32 | 11.5 / 11.7 / 11.9 | 11.90 / 12.10 / 13.09 |
| pooler m8 / m16 / m32 | 17.4 / 17.5-17.7 / 18.8 | 17.66 / 17.95 / 19.94 |
| unpooler m8 (1024 rows) / m16 / m32 | 4.4-4.7 / 17.9 / 18.1 | 5.12 / 18.40 / 20.22 |
| o m8 / m16 / m32 | 5.8 / 5.7 / 5.9 | 7.68 / 8.03 / 9.38 |

At m=128, in check `a97e1131`: pooler 73.1 against 22.5 and qkv_gate 41.0 against 17.1.

**Why:** Triton routes the decompressed BF16 through shared memory before the MMA (`local_alloc` and
one `convert_layout` per block in the TTGIR). `ncu` on the H100 shows 53 KB of shared memory,
4 CTAs/SM and issue in 18% of cycles: the kernel is latency-bound, not compute-bound.

## 2. c1 in Gluon with wgmma RS (weights in registers) and PTX decoding

Removes the round trip through shared memory: the c1 bytes are loaded in the layout of the wgmma A
operand, decoded in registers and fed to the MMA. There were four variants:
- RS: individual bytes;
- RS2 / rs2a: c1r format, with each thread's 32 bytes contiguous (`LDG.128`);
- rs4a: a 2-3 stage `cp.async` ring;
- rs5: rs4a with asynchronous wgmma.

The "a" variants decode with the packed PTX from `c1.py`, which brings the SASS down to 448-560
instructions from 984-1088.

First batch, without PTX (check `a97e1131`):

| shape | kf (v11) | Triton c1 MMA | RS | RS2 |
|---|---|---|---|---|
| pooler m8 / m16 / m32 | 17.31 / 17.25 / 19.39 | 17.22 / 17.41 / 25.31 | 32.0 / 31.7 / 36.0 | 26.1 / 26.9 / 29.0 |
| qkv_gate m8 / m16 / m32 | 11.65 / 11.65 / 12.13 | 12.96 / 13.09 / 16.13 | 22.0 / 22.2 / 24.0 | 19.2 / 19.5 / 20.1 |
| unpooler m16 / m32 | 17.54 / 17.73 | 18.21 / 26.14 | 32.4 / 36.9 | 27.1 / 29.6 |
| pooler m128 / qkv m128 | 22.53 / 17.12 | 73.1 / 41.0 | 97.8 / 71.6 | 46.3 / 37.3 |

With PTX (check `fedf62b9`):

| shape | kf (v11) | Triton c1 MMA | **rs2a c8** | rs4a c8s2 | best rs5 |
|---|---|---|---|---|---|
| pooler m1 / qkv m1 / o m1 / unp m1 | 15.23 / 10.24 / 5.86 / 3.81 | — | — | — | 18.43 / 13.22 / 8.96 / 6.53 (N=8) |
| pooler m8 / m16 / m32 | 17.63 / 17.89 / 18.91 | 17.54 / 17.76 / 25.22 | **17.28** / 18.24 / 21.22 | 19.4 / 20.4 / 22.5 | 18.75 / 19.52 / 22.30 |
| qkv_gate m8 / m16 / m32 | 11.58 / 11.49 / 12.29 | 12.90 / 12.93 / 15.62 | 11.87 / 12.22 / 13.60 | 12.9 / 13.7 / 15.5 | 13.41 / 13.98 / 15.58 |
| unpooler m8 / m16 / m32 | 4.38 / 17.15 / 17.60 | 4.99 / 17.89 / 25.86 | 6.05 / 18.40 / 21.60 | ≥ 6.3 | ≥ 6.7 / 20.0 / 22.6 |
| o m8 / m16 / m32 | 5.86 / 5.73 / 5.92 | 8.5 / 8.6 / 12.0 | 8.42 / 9.54 / 11.62 | ≥ 9.7 | ≥ 9.25 |
| m128 (pooler/qkv/unp/o) | 22.2 / 16.6 / 18.1 / 6.5 | — | — | — | 39-45 / 35 / 42 / 21 |

rs2a with 8-32 CTAs/SM (check `ec3537f2`) does not win in any shape either:
- at m=1, 16.7 against 15.3 on the pooler and 11.5 against 10.1 on qkv_gate;
- at m=8, 17.5 against 17.3;
- at m=16/32, o and m=128, worse.

**Why:** the best variant is the simplest one (rs2a: 4 KB of shared memory, 56 registers), and it
only ties on pooler m8. Pipelining (rs4a, rs5) makes it worse. With what Triton 3.8 offers, the
bottleneck is not the BF16 round trip through shared memory but the latency of a single warpgroup
that decodes and multiplies in series.

## 3. Single kernel without split for m=1

`_gemv_c1_direct`: applies the c1 correction inside the kernel and writes the BF16 output directly,
with no workspace and no PDL finish. Check `d3efb532`; "d wWpP(sS)" means W warps, P CTAs/SM and
split S:

| shape | kf (v11b; three measurements) | pure read c1 / BF16 | best single kernel | other variants |
|---|---|---|---|---|
| o m1 | 5.38 / 5.76 / 5.63 | 3.78 / 5.31 | 6.27 (d w4p3, s1) | 7.36-9.34 |
| unpooler m1 (1024 rows) | 3.71 / 3.65 / 3.84 | 2.40 / 2.98 | 16.38 (d w4p3, s8) | 23.9-30.2 |
| qkv_gate m1 | 10.05 / 9.98 / 10.02 | 7.90 / 10.24 | 33.41 (d w4p3, s4) | 47.9-61.0 |
| pooler m1 | 15.39 / 15.10 / 15.39 | 13.22 / 17.18 | 34.37 (d w4p3, s2) | 48.7-61.6 |

**Why:** without split, a kernel with few CTAs does not have enough bytes in flight to read W at HBM
rate. Split-K with a PDL finish costs less than what is lost this way.

## 4. Warp-specialized GEMM in Gluon against cuBLAS (m=128/256)

`gemm_ws.py`, `_ws_gemm`:
- swap-AB;
- one producer warp that loads W `[BN, 64]` and x `[BM, 64]` by TMA into a ring with mbarriers;
- consumer warpgroups with asynchronous wgmma;
- TMA epilogue: direct store with split 1, and with split > 1, a `cp.reduce.async.bulk` reduction
  plus a finish.

Best result (check `65e312f9`; the three qkv_gate measurements come from the same run):

| shape | kf / kf2 (v11b; cuBLAS in these shapes) | SGLang | best WS |
|---|---|---|---|
| qkv_gate m128 | 17.02 / 17.15 | 17.09 | **16.67-16.74** (n128m128k64s4 x7, three measurements) |
| pooler m128 | 21.89 / 22.46 | 25.25 | 24.10 (n128m128 x4); BN 256: 25.9 |
| qkv_gate m256 | 19.46 / 19.42 | 19.46 | 22.40 (n256m128 x7) |
| pooler m256 | 28.16 / 28.64 | 30.53 | 32.96 (n256m128 x4) |

Before the TMA epilogue (check `feb67807`, with per-element `red`), it lost by 1.5 to 2.5 times: for
example, o m128 11.97 against 6.27. Other custom GEMMs for m=128/256 that did not reach cuBLAS
either:
- **CuTe DSL** (checks `b5a1c1c7` and `447076fe`, about 30 configurations): wins only on the pooler,
  1.05-1.12x; elsewhere, 0.76-1.00x.
- **TMA + wgmma in Triton, without specialization** (check `fef78c63`): o m128 14.02 against 6.6;
  pooler m128 26.59 against 24.4; m256, 1.3 to 2.3 times worse.
- **DeepGEMM, cuBLASLt and FlashInfer** (shipped in the judge's image; check `6ffb77ba`): tie with
  cuBLAS.

**Why:** on the H200, cuBLAS (CUTLASS sm90 pingpong) already delivers what a hand-written BF16 GEMM
gets. WS ties on qkv_gate m128 (−0.4 µs, 2-3%) and loses 10-20% on the pooler and at m=256. Going
lower would require reading fewer bytes (c1 inside the WS GEMM). It was not done, because in the
layer those projections are 13-24% of their shape: ~+1% at most.

## 5. MoE lab with few tokens (check `af6d1b9c`, 28 Oct at 01:06)

The lab's table, as recorded. µs per call minus the empty graph
(4.29 µs), in a CUDA graph with cold L2. "bs (exp)" is the token count and, in parentheses, the
number of distinct local experts; two numbers separated by "/" are two repetitions in the same run.

| bs (exp) | v24 | **pdl** | pdl s4 | db | db s4 | fs | fs2 | fc | fs2 fc | cp2 | cp3 | cp4 | best k1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (2) | 33.82 / 33.60 | **31.71 / 31.65** | — | 32.42 / 33.12 | — | 32.13 / 32.10 | 32.96 | 32.26 / 31.87 | 35.04 / 34.05 | 40.00 | 38.59 | 46.66 | 32.00 (k1 c2 s4/4) |
| 1 (3) | 42.82 / 43.36 | **40.03 / 40.29** | — | 42.30 / 42.37 | — | 41.25 / 41.15 | 42.02 | 40.48 / 40.48 | 42.75 | 47.49 | 54.14 | 62.14 | 40.00 (k1db c3 s4/4) |
| 8 (7) | 71.14 / 71.30 | **68.19 / 68.35** | 72.03 | 71.10 | 75.26 | 80.54 | 72.74 | 73.02 | 77.95 | 90.27 | 99.17 | 125.44 | — |
| 8 (5) | 61.22 / 61.38 | 59.36 / 59.49 | **58.30** | 60.93 | 62.18 | 77.22 | 65.92 | 63.94 | 70.91 | 66.85 | 81.09 | 96.83 | — |
| 16 (9) | 94.27 / 94.08 | 93.38 / 93.41 | **90.30** | 96.19 | 93.70 | 127.84 | 100.42 | 103.52 | 110.11 | 106.40 | 127.58 | 158.62 | — |
| 32 (12) | **115.49 / 115.97** | 118.43 / 118.43 | 119.20 | 128.06 | 129.66 | 183.14 | 135.39 | 133.15 | 148.35 | 134.08 | 156.80 | 215.68 | — |

Variants:
- **pdl:** the PDL chain over the five kernels.
- **s4:** split-K 4 for bs > 6.
- **db:** double buffering by half groups.
- **fs / fs2:** SwiGLU in the gemm1 epilogue, with atomics / without them.
- **fc:** sum of squares in gemm2 and a gridded combine.
- **cpS:** per-warp `cp.async` ring of S groups.
- **k1:** the whole MoE of one token in one cooperative kernel (`moe_bs1`).

Only the PDL chain wins, up to bs 16. At bs 32 it loses ~2.9 µs, hence `PDL_MAX_BS = 16` in
[`experiments/small_m_clean.py`](experiments/small_m_clean.py). pdl s4 has a
mixed sign depending on the routing (a single measurement). Everything else loses or ties.
