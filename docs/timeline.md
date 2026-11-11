# Timeline

Times are CET. Ranks come from the snapshots in [`evidence/boards/`](../evidence/boards/) and from
the state document kept during the work. The board keeps growing: losing rank with the same mark is
normal. [`charts/rank_by_day.png`](../charts/rank_by_day.png) plots it.

## 24 Oct: templates and calibration

The templates are downloaded with `okc` 0.7.1 and the plan is written. Stated goal:
top 50 overall, with the top 10 as a stretch. Shortly before midnight UTC, five checks of the
untouched baseline go out, one per problem, using the day's quota before it expired. They are the
H200 half of the calibration.

## 25 Oct: block 1, the factory

- **The bench.** The judge's image (16.6 GB) is built locally and uploaded to the cluster in ~3 h,
  limited by the home uplink. First worker on an H100 at 05:13. The five baselines pass their
  tests, and the A/A test gives 0.998-1.001.
- **Calibration.** The per-shape H200/H100 factor comes out between 0.40 and ~1.0
  ([`method.md`](method.md)).
- **First submit.** bf16 enters at 1.207x (`4eb830e7`), rank 380.
- **4 of 4 problems on the board** that afternoon. First submits: attention 1.300x, MoE 1.068x,
  decode_layer 1.221x. Attention quickly climbs to 1.547x and 1.588x, all in Triton.
- **Incident.** The worker in the interactive QoS lands on a node without a GPU and swallows runs
  for hours; about 2 h of bench time are lost.

## 26 Oct, into the early hours of the 27th: block 2, the big gains

- **MoE.**
  - 1.220x with the GEMMs in CUDA (v3).
  - 1.367x with routing in CUDA and 4 CTAs/SM (v6).
  - **1.606x** with the reordered bits (v8, +17%).
  - 1.709x with recoded scales and item ordering (v12).
- **Attention.** **1.652x** with `full_fa2` in CUDA and 1.691x with `verify_ws`, which uses
  specialized warps.
- **bf16.** 1.231x (v9, PDL closing kernel) and 1.261x (v10, lossless compressed weights).
- **decode_layer.** From 1.25x to **1.587x** in eight submits, as the other problems' kernels come in.
- **Overall: rank 320 of 1190** at 09:00 on the 26th, and 299 of 1266 at the end of the block
  (27 Oct, 02:40). That day work stopped to wait for quota and ~10 h were lost.

## 27 Oct: block 3, the layer

- **MoE v24, 1.811x** (IMAD in the decode, `int` indices, addresses with immediate offsets).
- **decode_layer.**
  - 1.650x and 1.658x with MoE v24, the round-7 attention and bf16 v11b.
  - **1.675x (`b76b7763`, v45)** with the PDL chain for the small-batch MoE. This is the layer's
    final mark.
- **Bench lesson.** The first cases where the H100 bench gives the wrong sign show up that day:
  wgmma in MoE, c1 with 32 tokens, `CLW` and `PVA` in attention. From then on the H200 check makes
  the call.
- **The overall board follows decode_layer.** I verify that it ranks exactly like that board.
- **Overall: rank 291 of 1352** at 22:05.

## 28 Oct: last iterations of block 3

- **Attention 1.760x** (`f31f6572`): `cl` with clusters and DSMEM for sliding at batch 1, plus a
  one-pass combine.
- **MoE.** The overnight batch (specialized warps, wgmma with a queue, v38) does not beat v24.
- Work stops at 02:10.

## 29 and 30 Oct: pause and final batch

- **No work on the 29th.** On the 30th at 19:30, the overall board has me at **rank 513 of 1711**:
  the board has grown and moved up.
- **The final batch.** One more short batch before building the repo, with a limit of 3.5
  H100-hours, of which 0.23 are spent.
  - **MoE v55, 1.8585x** (`db79106e`): dynamic per-warp item queue, +2.6%.
  - **Attention s4, 1.801x** (`28dd5b80`): sliding verify through `verify_ws`, +2.3%.
  - decode_layer v53 and v54 stay inside the noise, with no submit. bf16 gets no more work.
- **End of experiments:** 23:07 on the 30th.

## 1 to 4 Nov: wind-down

- **1 Nov.** The batch is closed: overall **605 of 1810**, with the same marks.
- **2 Nov.** This repo is exported. Overall **656 of 1873**.
- **4 Nov.** Freeze at 20:00 CET.

## 5 Nov: close

The kernel track closed on 5 Nov 2026 at 08:59 CET. I made no submissions after 30 Oct, so the
final speedups are the ones from the final batch.

| problem | final speedup | rank on 2 Nov | final rank |
|---|---|---|---|
| overall | | 656 / 1873 | 748 / 2014 |
| muse_decode_layer | 1.675x | 650 / 1196 | 745 / 1300 |
| muse_moe_w4a16 | 1.859x | 662 / 1319 | 756 / 1426 |
| muse_attention_decode | 1.801x | 668 / 1267 | 759 / 1380 |
| muse_bf16_proj | 1.261x | 790 / 1651 | 819 / 1747 |

My speedups did not change between 2 Nov and the close. The ranks fell because late teams entered
the boards: the overall board grew by ~140 teams (from 1873 to 2014). At the close, the top 40 was
at 2.418x on decode_layer, 2.563x on MoE, 2.846x on attention and 1.611x on bf16.

## H100 hours

| day | GPU-hours |
|---|---|
| 25 Oct | 6.00 |
| 26 Oct | 6.03 |
| 27 Oct | 6.07 |
| 28 Oct | 0.41 |
| 30 Oct | 0.23 |
| **total** | **18.74** (60 jobs) |

The per-job detail is in [`evidence/h100-hours.jsonl`](../evidence/h100-hours.jsonl), taken from
`sacct`.
