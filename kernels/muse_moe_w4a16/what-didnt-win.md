# MoE NVFP4: what didn't win (H200 numbers)

Every number is the judge's speedup (H200, CUDA graph, cold L2) against SGLang, from an `okc check`
with 5/5 tests. Each row is compared with the variant it was built on ("base" column). The evidence
is in [`evidence/runs/muse_moe_w4a16/`](../../evidence/runs/muse_moe_w4a16/)`<check>.{json,txt}`.

**What did win, for context** (bs 128 and bs 256 are verify 16 and verify 32, 75% of the MoE time
in the layer):

| variant | check | total | bs 128 | bs 256 | what |
|---|---|---|---|---|---|
| v3 | `d580b17d` | 1.225 | | | CUDA GEMMs with `mma.sync`, swap AB and no shared memory |
| v8 | `5f008f07` | 1.601 | | | reordered bits: decode from 15 to 11 operations per word |
| v12 | `5ed6b34c` | 1.709 | 1.866 | 1.798 | item order with the weight tile as the fastest index |
| v23 | `983b4750` | 1.768 | 1.949 | 1.860 | decode with one IMAD (10 operations per word) |
| v24 | `f1dbbda7` | 1.812 | 2.018 | 1.980 | 64-bit addresses per group, scale bias in a register (submit `fc6aaf56`) |
| **v55** | `4796ee91`, `08bd5408`; submit `db79106e` | **1.863** (submit 1.8585) | 2.095 | 2.031 | PDL base up to bs 16 + dynamic per-warp item queue (mean of two checks) |

## What didn't win

| family | variant | check | total | bs 128 | bs 256 | base (total) | why, according to the measurement |
|---|---|---|---|---|---|---|---|
| **overlapping loads / rings** | v4, software pipeline | `b9b31529` | 1.183 | | | v3 (1.225) | +40 registers per thread and 3 CTAs per SM; ptxas already hoisted the loads in the unrolled loop |
| | v14, per-warp `cp.async` ring | `a874feed` | 1.634 | 1.824 | 1.794 | v12 (1.703) | smaller weight bursts per warp |
| | v15a, ring 8 × 3 CTAs | `f0dc6e44` | 1.441 | 1.611 | 1.593 | v12 | fewer CTAs per SM |
| | v15b, ring 6 × 4 CTAs | `d492785c` | 1.531 | 1.701 | 1.675 | v12 | same |
| | v22, rolling prefetch | `8cc92dd6` | 1.587 | 1.761 | 1.670 | v20 (1.725) | 4 KB in flight continuously instead of 8 KB bursts |
| **32-token blocks** | v19a, BM 32 from bs 65 | `00a7a987` | 1.682 | 1.763 | 1.797 | v12 (1.709) | −43% instructions per token in full blocks, but more registers and more activation loads |
| | v19c, BM 32 from bs 129 | `a0962828` | 1.713 | 1.870 | 1.803 | v12 | tie |
| | v34a, BM 32 in 128 registers | `c9c896df` | 1.779 | 1.853 | 1.903 | v24 (1.812) | half groups: smaller bursts |
| | v34c, same from bs 129 | `f0142dc7` | 1.804 | 2.043 | 1.886 | v24 | same |
| **work split and fusion** | v13, 8-warp CTAs | `303f5e98` | 1.689 | 1.822 | 1.847 | v12 | 2 CTAs per SM |
| | v18, split-K SwiGLU fused | `31b3228b` | 1.569 | 1.778 | 1.740 | v17 | instructions in the hot loop |
| | v25, gemm2 split × 2 | `40f13368` | 1.809 | 2.022 | 1.982 | v24 | tie |
| **MMA chains** | v21, two MMA chains | `c7900102` | 1.716 | 1.858 | 1.780 | v20 (1.725) | tie: the MMA is not the limit |
| | v26, v24 + two chains (NT = 1) | `c72c205c` | 1.824 | 2.076 | 1.978 | v24 | tie |
| **contiguity** | v35, interleaved tiles of a pair (`_pair_tiles`) | `1ca974e2` | 1.818 | 2.037 | 1.988 | v24 | tie: 8 KB contiguous against 2 × 4 KB changes nothing |
| **metadata** | v38, next-item table fetched ahead | `d312df1c` | 1.823 | 2.032 | 2.002 | v35 (1.818) | tie |
| | v56, v55 + v38 (next item reserved) | `c5f4c610` | 1.831 | 1.989 | 1.973 | v55 (1.863) | reserving one item ahead takes balance away from the queue |
| **wgmma** (decode A in registers, B in shared) | v41, both GEMMs, N = 32 | `afe582d7` | 1.733 | 1.641 | 1.839 | v35 | on the bench −18% at verify 32; on the H200, items 4 times coarser (one CTA) |
| | v45, N by block | `39964df0` | 1.622 | 1.412 | 1.429 | v41 | two kernels per GEMM |
| | v46, `mma.sync` for blocks ≤ 16 | `d69ce9c9` | 1.684 | 1.586 | 1.591 | v41 | same |
| | v47, wgmma with split-K 2 | `84b8b790` | 1.726 | 1.647 | 1.806 | v41 | tie with v41 |
| | v48, dynamic per-CTA queue | `a6ae6aab` | 1.757 | 1.710 | 1.917 | v41 | better than v41 in both shapes: the queue lead (→ v55) |
| **warp specialization** (producer decode → shared, consumer wgmma, mbarriers) | v51, specialized gemm1 | `bb8909ba` | 1.497 | 1.074 | 1.157 | v35 | gemm1 324-446 µs against 155-197: 8 warps per SM decode instead of 16 |
| | v52, + 16-entry table for gemm2 | `9904a31a` | 1.513 | 1.128 | 1.189 | v51 | gemm2 back to v24 time; gemm1 unchanged |
| | v53, + `cp.async` ring of raw weights | `d0e29b9d` | 1.515 | 1.144 | 1.198 | v52 | the producer was not waiting on memory |
| | v54, gemm2 specialized too | `a824d7e8` | 1.418 | 0.921 | 0.984 | v53 | gemm2 177-234 µs against 74-96 |
| **perturbation** (diagnostic) | P1, decode × 2 | `8f0299bf` | 1.087 | 1.085 | 1.060 | v20 | proof that the decode is the limit on the H200 |

## Lessons

1. **On the H200 the GEMMs are limited by decode issue rate and latency, not by memory.** What
   removed decode instructions won (v8, v9, v23, v24), and what doubled them sank the total (P1).
2. **Fewer decoding warps loses, even if everything else gets faster.** Confirmed by 8-warp CTAs
   (v13), rings with 3 CTAs (v15a), per-CTA wgmma (v41-v47) and warp specialization (v51-v54).
3. **Splitting a warp's 8 KB burst loses.** v14, v15, v22 and v34: four times.
4. **The bench (H100) predicts none of this.** wgmma and warp specialization won or tied on the
   H100, which runs its DRAM at ~90%, and lost on the H200. The bench is good for correctness,
   not for timing changes to the decode, the MMA or the work split.
5. **The last thing left was the work split.** The dynamic queue gave +3.8% and +2.6% at verify
   16/32 (v55): items with `cnt > 8` (two token tiles) cost more than single-tile items, and the
   static split left the last round to the slow warps.
   **It does not show in the layer:** with the `muse_decode_layer` routing (dist soft, spread
   agentic), decode_layer measured v55 in its v54. Check `54914d08`: 1.6684x against 1.6663,
   noise. gemm1 wins at verify 16 sliding (138.5 → 122.9 µs) and ties or loses 1-3% at verify 32.
   What the queue gains depends on how much item cost varies with the routing.

## Final result

**Submit `db79106e` of v55: 1.8585x** [1.831, 1.886], rank 538 of 1203 (v24 was at 1.811x).
Two notes for whoever picks it up:
- `nblk_ptr` is `const int* __restrict__` and `gemm()` writes it through `const_cast`. It should be
  `int* __restrict__`.
- The 95% CIs of v55 and v24 barely overlap: +2.6% on the board.

## Variants worth keeping as examples

1. **`submission.py`** (readable version: [`kernel.py`](kernel.py)). The complete final file: the
   bit-level NVFP4 decode (`pack_codes`, `_decode_source` and the IMAD from `_imad_form`),
   `mma.sync` with swap AB and no shared memory, split-K, the PDL chain up to bs 16 and the
   per-warp dynamic queue. It is the file handed to the layer.
2. **[`versions/v24_fc6aaf56.py`](versions/v24_fc6aaf56.py).** The same GEMM without PDL or queue,
   123 lines shorter. It is the best point to read the kernel from, and it is also submit
   `fc6aaf56`.
3. **[`experiments/v41_wgmma_both.py`](experiments/v41_wgmma_both.py).** wgmma with A in
   registers (the fragment the decode already produces) and B in shared memory without swizzle: a
   minimal example of `wgmma.mma_async` RS, together with
   [`diagnostics/wgmma_micro.py`](diagnostics/wgmma_micro.py), its micro-test against float64.
4. **[`experiments/v54_ws_gemm2.py`](experiments/v54_ws_gemm2.py).** Full warp specialization in
   both GEMMs: `full`/`empty` mbarriers with parity, `setmaxnreg` 104/152, `stmatrix` and wgmma SS,
   plus its emulated version for sm75. It works and is correct, even though it loses on this
   problem. It goes with [`diagnostics/ws_micro.py`](diagnostics/ws_micro.py).

The tools that explain how the decode was reached are
[`tools/search_bit_layout.py`](tools/search_bit_layout.py) (v8) and
[`tools/search_imad_layout.py`](tools/search_imad_layout.py) (v23).
