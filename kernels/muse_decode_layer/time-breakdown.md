# Breakdown of the layer's time (H200)

decode_layer, close of 30 Oct. Source: the H200 profile of the final check, `54914d081d7c498c`
(v54, `3b35898`, sha256 `7aa4a1a1…`, 1.6684x [1.643, 1.693]): v53 with the MoE kernel's single
experts file (v55, dynamic work-item queue). The board mark is v45 (submit `b76b7763`, 1.6754x);
v45, v53 and v54 are within the noise of one run (±1.5%), so the breakdown holds for all three. It
is regenerated with `python3 kernels/muse_decode_layer/tools/breakdown.py <job>`
([`tools/breakdown.py`](tools/breakdown.py)).

**How to read it.**
- Each okc profile gives the 12 heaviest kernels of a shape and the GPU total; the rest goes to
  "others". With 128-256 tokens `route_pre_expert` falls outside the 12 and lands in "others".
- The GPU time adds up both streams (the router and the routing overlap with the pooler), and with
  PDL a kernel counts as its own the time it waits in `griddepcontrol.wait`, so it exceeds the wall
  time. That is why each shape's breakdown is computed on GPU time and applied to the measured wall
  time.
- The score is the geometric mean of six loads; each load adds its two shapes (sliding ×46 and full
  ×16), and all of them count the same.
- The floors come from the tables in the `TASK.md` of each standalone problem (H200, 4.8 TB/s or
  989 TFLOP/s):
  - projections: qkv + o + pooler + unpooler (the 1024-row slice up to 8 tokens) + router, from
    `muse_bf16_proj`;
  - MoE: from `muse_moe_w4a16` with bs = tokens;
  - attention: from `muse_attention_decode`, whose full shapes use `dist coherent` while the layer
    uses `soft` (same number of bytes).

Components by kernel name:
- **attention:** `_attn_kernel`, `_combine_kernel`, `full_fa2`, `verify_ws`;
- **MoE:** `moe_gemm*`, `moe_route`, `_route_prep`, `_swiglu_reduce`, `_combine`;
- **proj. c1:** `_c1_*` (qkv with 1 token through stream-K; pooler 1-16 tokens; o and the unpooler
  slice with 1 token; the full unpooler with 9-16);
- **proj. split-K:** `_proj_kernel*` and `_splitk_finish` (qkv 2-32 tokens, pooler 17-32, the
  unpooler slice 2-8, router always);
- **proj. cuBLAS:** `nvjet*` (o from 2 tokens; the unpooler from 17; everything from 33);
- **routing + pre-expert:** `_route_pre_expert_kernel`;
- **boundaries:** SGLang's after the attention, and `_ffn_boundary_kernel` (with the unpooler
  finish inside) up to 8 tokens; above that, both from SGLang.

### Per shape (wall µs, split by each component's GPU time)

| shape | SGLang | mine | speedup | layer floor | attention | MoE | proj. c1 | proj. split-K | proj. cuBLAS | routing + pre-expert | boundaries (norm + residual) | others (outside the top 12) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode 1 sliding | 171.3 | 92.8 | 1.847 | | 6.5 | 33.8 | 35.6 | 3.8 | 0.0 | 3.2 | 7.5 | 2.4 |
| decode 1 full | 180.6 | 100.1 | 1.805 | | 12.7 | 29.1 | 38.3 | 4.0 | 0.0 | 3.0 | 8.1 | 4.8 |
| decode 16 sliding | 259.6 | 157.4 | 1.649 | | 7.7 | 90.0 | 30.6 | 12.1 | 5.3 | 2.8 | 3.0 | 5.9 |
| decode 16 full | 435.7 | 250.5 | 1.739 | | 96.9 | 78.9 | 37.9 | 15.1 | 6.7 | 0.0 | 7.2 | 7.8 |
| decode 32 sliding | 263.5 | 187.8 | 1.403 | | 14.5 | 91.9 | 0.0 | 39.6 | 26.3 | 3.3 | 7.6 | 4.5 |
| decode 32 full | 611.8 | 366.1 | 1.671 | | 194.5 | 90.2 | 0.0 | 38.1 | 27.0 | 3.3 | 8.0 | 5.0 |
| verify 1 sliding | 265.0 | 133.9 | 1.979 | | 6.8 | 67.7 | 16.9 | 19.2 | 6.5 | 3.2 | 7.3 | 6.4 |
| verify 1 full | 272.1 | 142.4 | 1.911 | | 12.4 | 79.3 | 13.9 | 16.5 | 5.4 | 2.6 | 6.4 | 6.0 |
| verify 16 sliding | 482.3 | 302.8 | 1.593 | | 13.6 | 191.0 | 0.0 | 6.3 | 70.5 | 0.0 | 9.5 | 12.0 |
| verify 16 full | 676.8 | 465.0 | 1.455 | | 174.8 | 186.2 | 0.0 | 6.6 | 75.2 | 0.0 | 9.6 | 12.5 |
| verify 32 sliding | 678.1 | 427.8 | 1.585 | | 23.0 | 289.8 | 0.0 | 8.0 | 82.6 | 0.0 | 15.1 | 9.3 |
| verify 32 full | 1050.7 | 735.0 | 1.429 | | 320.5 | 292.5 | 0.0 | 10.0 | 86.5 | 0.0 | 15.3 | 10.2 |

### Per load: weight of each component in the load's time, and floors

| load | speedup | attention | MoE | proj. c1 | proj. split-K | proj. cuBLAS | routing + pre-expert | boundaries (norm + residual) | others (outside the top 12) | proj. floor | MoE floor | attention floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode 1 | 1.836 | 9% | 34% | 38% | 4% | 0% | 3% | 8% | 3% | 28.9 | 11.4 | 1.2 (weighted mean) |
| decode 16 | 1.681 | 17% | 48% | 18% | 7% | 3% | 1% | 2% | 4% | 41.5 | 52.5 | 17.5 (weighted mean) |
| decode 32 | 1.511 | 26% | 39% | 0% | 17% | 11% | 1% | 3% | 2% | 41.7 | 70.8 | 35.1 (weighted mean) |
| verify 1 | 1.960 | 6% | 52% | 12% | 14% | 5% | 2% | 5% | 5% | 29.0 | 37.3 | 1.2 (weighted mean) |
| verify 16 | 1.545 | 16% | 55% | 0% | 2% | 21% | 0% | 3% | 4% | 43.8 | 101.3 | 17.8 (weighted mean) |
| verify 32 | 1.527 | 20% | 57% | 0% | 2% | 16% | 0% | 3% | 2% | 51.8 | 108.9 | 35.5 (weighted mean) |

Geometric mean of this check: 1.6684x.

### Average weight in the score (mean over the six loads, each counts the same)

| component | average weight | gain if it were halved |
|---|---|---|
| attention | 15.6% | x1.085 |
| MoE | 47.6% | x1.315 |
| proj. c1 | 11.3% | x1.063 |
| proj. split-K | 7.5% | x1.039 |
| proj. cuBLAS | 9.4% | x1.050 |
| routing + pre-expert | 1.4% | x1.007 |
| boundaries (norm + residual) | 4.1% | x1.021 |
| others (outside the top 12) | 3.1% | x1.016 |

### Scenarios: geometric mean if each part reached k times its floor (the rest as measured)

| scenario | geometric mean | over this check |
|---|---|---|
| measured | 1.668x | x1.000 |
| MoE at 2 x floor | 1.746x | x1.046 |
| MoE at 1.5 x floor | 1.889x | x1.132 |
| MoE at the floor | 2.171x | x1.301 |
| attention at 2 x floor (min. 5 µs) | 1.723x | x1.033 |
| attention at the floor (min. 5 µs) | 1.805x | x1.082 |
| projections at 1.3 x floor | 1.732x | x1.038 |
| projections at the floor | 1.834x | x1.099 |
| small kernels at 4 µs per shape | 1.783x | x1.069 |
| MoE 2x + attention 2x + proj. 1.3x | 1.883x | x1.129 |
| MoE 1.5x + attention 2x + proj. 1.3x + small 4 µs | 2.231x | x1.337 |
| everything at the floor | 3.110x | x1.864 |

## Reading

- **The MoE is 48% of the score**, and 52-57% in verify 1/16/32. Against its floor: verify 32 at 2.7
  times (~290 µs against 109), verify 16 at 1.8-1.9, verify 1 at 1.8-2.1, decode 16 at 1.5-1.7 and
  decode 32 at 1.3. In decode 1 the profile gives 29-34 µs (2.6-3.0 times), but there the PDL chain
  makes gemm1 count the wait on its primary as its own, so the number is inflated. On its own,
  brought to 1.5 times its floor, it gives **x1.132**.
- **Attention, 15.6%**: the full part. verify 16/32 full runs at 2.8 and 2.5 times its floor (175
  and 320 µs), decode 16/32 full at 1.5. The sliding layers are 6.5-23 µs. The `f31f6572` attention (v52 onwards)
  brought decode 1 full down from ~107 to ~100-102 µs of wall time. At twice the floor it gives
  x1.033; at the floor, x1.082.
- **Projections, 28% across the three paths** (c1 11%, split-K 8%, cuBLAS 9%): ~1.4 times the floor
  with few tokens and ~2 times with 128-256 (cuBLAS, close to compute in pooler and unpooler;
  qkv_gate is the weak one). At 1.3 times the floor they give x1.038.
- **Small kernels** (boundaries, routing, finishes, "others"): ~9% of the time. Bringing them to
  4 µs per shape would give x1.069, but only with a megakernel with grid barriers, which was estimated
  at ~+1% net.
- **MoE v55 in the layer.** In its own problem it beats v24 on every shape (check 1.863x against
  1.812x). In the layer, against v53, it only shows in verify 16 sliding (gemm1 138.5 → 122.9 µs)
  and verify 1 (gemm1 41.5 → 36.5). With 256 tokens it ties or loses a little (gemm1 188-198 →
  194-201). The layer's routing (dist soft, spread agentic) is not the one in its problem.

## What rank 40 would take (2.252x on 30 Oct; x1.344 over the board mark)

No single piece is enough. "MoE at 1.5 times its floor + attention at 2 times + projections at 1.3
times + small kernels at 4 µs" gives 2.231x, still below. Without the small kernels, "MoE 2x +
attention 2x + projections 1.3x" gives 1.883x; rank 300 (1.992x) already asks for more than that.

Order of impact by piece:
1. **MoE** (the one that moves the most): verify 32 (~290 µs, 2.7 times its floor) and verify 16
   (~190 µs).
2. **Attention**: verify 16/32 full (175 and 320 µs).
3. **Projections**: a custom GEMM for qkv_gate with 128-256 tokens; c1 with few tokens is already
   close to its floor.
4. **Layer**: the integration of each piece. The small kernels, the PDL chain and the streams give
   nothing more without a design change.

## H100 bench against the H200: what the bench can judge (27 Oct, 17:30)

The bench ([`scripts/bench/bench`](../../scripts/bench/bench), the H100 of the university cluster) **cannot judge changes that trade bytes
for compute, nor changes of MMA path**: twice today it gave the opposite sign from the judge's
(H200).

| change | H100 bench | H200 check |
|---|---|---|
| MoE v41: wgmma in both GEMMs from 65 tokens | verify 32 −18% (MoE alone); layer x1.250 against x1.217 | with 128 tokens, gemm1 112 → 167 µs and gemm2 61 → 90 µs; layer 1.634 against 1.669 (`2e7740f3`) |
| c1 (compressed BF16) with 17-32 tokens, and qkv_gate in c1 with 2-32 | decode 32 −15/−12 µs, decode 16 −6 µs | decode 32 +12/+31 µs, decode 16 +8 µs; layer 1.653 against 1.676 (`798dd690`) |

The H200 has about three times the bandwidth of the bench's 64 GB H100 (4.8 against ~1.6 TB/s
measured) with similar compute. What hides behind memory on
the H100 (decompressing, decoding, changing the MMA) becomes the bottleneck on the H200.

**Rule.**
- The bench is good for the tests and for changes that only remove latency or bytes. On those it
  has been right: stream-K for qkv with 1 token, the finish fused into the boundary, the per-weight
  correction limit.
- Compute or MMA-path changes need evidence on the H200 (a check, or the H200 numbers of the
  problem that owns the kernel) before they are stacked or submitted. Nothing of that kind is stacked on
  bench gains.

The other problems showed the same (see [`../muse_attention_decode/what-didnt-win.md`](../muse_attention_decode/what-didnt-win.md)
and [`../muse_moe_w4a16/what-didnt-win.md`](../muse_moe_w4a16/what-didnt-win.md)):
- attention, verify full: whole 128 B rows per cluster (`CLW`), bench −30% and H200 +18%; PV in FP8
  with wgmma, bench +1.5% and H200 +45%;
- MoE (28 Oct), specialized warps in the NVFP4 GEMMs (v51-v54): a tie on the bench, ~2 times slower
  on the H200 (only 8 warps per SM decode, against 16 in v24).
