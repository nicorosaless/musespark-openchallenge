# Attention gap to rank 40 (2.52x), 2026-10-27

Data from submit `d0139e27978b4c49` (H200, 1.6872x). The µs are the judge's scored median (with the
fixed cost of the graph already subtracted), and the kernels come from the run's own profile.

**H200 floors:**
- bytes: 4.8 TB/s peak. The reads are the E4M3 KV cache (2 heads × 64 B of K and as many of V per
  key), q, k, v and the output in bf16, and the page entries (4 B per key);
- compute: 989 TFLOP/s in dense BF16 and 1979 in FP8 (4·keys·q_len·16·64 FLOP).

**Score.** It is the geometric mean of six loads. Each load weighs 46 sliding layers and 16 full
ones, with the weighted sum of times ([`breakdown.py`](tools/breakdown.py)).

## 1. Per shape: my time against the floor

| load | layer | SGLang µs | my µs | kernels (GPU µs) | bytes floor | FP8 floor | ×floor |
|---|---|---|---|---|---|---|---|
| decode 1 | sliding | 20.5 | 9.1 | `full_fa2` cl 7.7 | 0.11 | 0.00 | 80 |
| decode 1 | full | 31.0 | 20.0 | `full_fa2` 12.9 + combine 4.5 | 4.3 | 0.2 | 4.6 |
| decode 16 | sliding | 26.5 | 12.4 | Triton 7.7 + combine 2.1 | 1.8 | 0.1 | 6.8 |
| decode 16 | full | 200.6 | 109.6 | `full_fa2` 108.1 + combine 2.8 | 69.4 | 2.7 | 1.6 |
| decode 32 | sliding | 31.4 | 16.9 | Triton 11.4 + combine 1.9 | 3.6 | 0.1 | 4.7 |
| decode 32 | full | 386.6 | 201.4 | `full_fa2` 201.3 + combine 2.5 | 138.7 | 5.3 | 1.5 |
| verify 1 | sliding | 23.0 | 10.6 | Triton 5.5 + combine 2.0 | 0.13 | 0.03 | 84 |
| verify 1 | full | 35.4 | 20.9 | `verify_ws` 14.3 + combine 4.8 | 4.4 | 1.3 | 4.8 |
| verify 16 | sliding | 29.2 | 17.4 | Triton 11.8 + combine 3.4 | 2.0 | 0.5 | 8.6 |
| verify 16 | full | 204.7 | 169.7 | `verify_ws` 169.3 + combine 3.8 | 69.6 | 21.2 | 2.4 |
| verify 32 | sliding | 34.4 | 24.1 | Triton 16.4 + combine 4.7 | 4.0 | 1.1 | 6.0 |
| verify 32 | full | 390.8 | 322.1 | `verify_ws` 320.4 + combine 5.2 | 139.2 | 42.4 | 2.3 |

The large-batch full shapes are close to their bytes floor: decode at 1.5-1.6x and verify at
2.3-2.4x. All the sliding shapes and the batch-1 shapes are far from any floor (4.6-84x): they are
latency. What counts there is the launch, the serial DRAM hops (`seq_lens` → page table → K/V) and
the merge of the splits. Within each load's weight, the sliding layers are 56-59% of the time at
batch 1 and 18-25% at batch 16/32.

## 2. What 2.52x requires

If the gain were spread as in my profile, every shape would have to run
2.52 / 1.687 = 1.494 times faster:

| load | layer | mine | at 2.52x | ×floor at 2.52x |
|---|---|---|---|---|
| decode 1 | sliding | 9.1 | 6.1 | 54 |
| decode 1 | full | 20.0 | 13.4 | 3.1 |
| decode 16 | sliding | 12.4 | 8.3 | 4.6 |
| decode 16 | full | 109.6 | **73.4** | **1.06** |
| decode 32 | sliding | 16.9 | 11.3 | 3.1 |
| decode 32 | full | 201.4 | **134.9** | **0.97** |
| verify 1 | sliding | 10.6 | 7.1 | 56 |
| verify 1 | full | 20.9 | 14.0 | 3.2 |
| verify 16 | sliding | 17.4 | 11.6 | 5.8 |
| verify 16 | full | 169.7 | 113.6 | 1.6 |
| verify 32 | sliding | 24.1 | 16.1 | 4.0 |
| verify 32 | full | 322.1 | 215.6 | 1.5 |

**Rank 40's decode 16/32 full cannot look like mine.** At a uniform 2.52x they would have to run at
94-103% of peak HBM bandwidth, and not even a sequential read gets there; my random gather runs at
~65%. So rank 40 does not win by spreading the gain as I do: it has to win much more where I am
far from the floor. Scenarios on my geometric mean ([`breakdown.py`](tools/breakdown.py) with the data
above):

| scenario | total |
|---|---|
| today | 1.687 |
| verify full at the rate of decode full (×0.64) | 1.889 |
| sliding at a 4 µs floor (the rest unchanged) | 2.179 |
| all full shapes at their bytes floor (impossible) | 2.627 |
| full at 1.3x their floor and sliding at 5 µs | 3.173 |

The sliding layers are the lever. They weigh 46 of every 62 layers and sit at 5-84x their floor.
Just bringing them down to ~4 µs takes the total from 1.69 to 2.18. With these weights, rank 40's
2.52x looks like sliding at 4-6 µs and full at 1.3-1.6x the bytes floor.

## 3. What design explains sliding at 4-6 µs

What I do today on the sliding layers (2048 keys per request):
- **batch ≤ 2, decode:** one `full_fa2` kernel in clusters of 16 that merges through DSMEM (`cl`);
  7.7 µs of GPU time. With `TSTAMP` (bench): prologue of q and first pages 3.3-3.8 µs, first block
  +0.8, loop +0.2, epilogue +1.4 and cluster barrier up to 7.6;
- **the rest:** the Triton kernel A with split-KV and a second combine kernel (with PDL). The
  combine costs 1.9-4.7 µs of GPU time plus its launch.

What the reference kernels do in decode and verify with GQA 8 and small pages:
- **FA3** (`flash_attn_with_kvcache`, the judge's reference):
  - `pack_gqa`: the 8 q heads of the group (× q_len) go in the M dimension, and each K/V row is
    read once per group (as I do);
  - split-KV with a `num_splits` heuristic and a separate combine kernel;
  - with 1-token pages it does not use TMA: the gather uses `cp.async`;
  - on sliding it runs at 20-35 µs: it loses on latency, not on bytes.
- **FlashInfer**:
  - in decode with GQA ≥ 4, the prefill kernel with tensor cores (`use_tensor_cores`);
  - **plan/run**: the plan splits the keys of all requests into equal-size chunks across all the
    CTAs of **a single wave**, without tying a CTA to a request;
  - the state merge (`merge_states`) is a separate kernel, except when everything fits in one CTA;
  - in verify with small pages, the page entries of each chunk are loaded into smem before the
    copies.
- Neither of them gets below ~2 launches or below the three serial DRAM hops. 4-6 µs per sliding
  layer requires:
  1. **a single kernel** (no separate combine), with the merge inside;
  2. **one whole, balanced wave**: every SM with few blocks, so that each CTA's chain is a single
     K/V hop;
  3. **a shrunken prologue**, overlapping what does not depend on `seq_lens`: q and its
     RMSNorm/RoPE, and the write of the new k/v, in parallel with `seq_lens` → page table.

What I have: point 1 only at batch ≤ 2 (`cl` with clusters of 16). At batch 16/32 I still use two
kernels. `cl` at batch 16/32 was discarded on the bench because clusters of 16 do not fit in one
wave. But that measurement came from the bench, which I have already seen ranks things the other
way round, and a smaller cluster sized to one wave was never tried: 8 at batch 16 and 4 at batch 32
(128 CTAs on 132 SMs).

## First hypothesis to measure (today, with the 6 checks left)

**H1: sliding at batch 16/32 in a single kernel (`cl`) with the cluster sized to one wave.**
Clusters of C = 2^⌊log2(132 / (batch · pairs))⌋, at most 16:
- decode 16 / 32 (KVP 2, 1 pair): C = 8 / 4;
- verify 16 / 32 (KVP 1, 2 heads): C = 4 / 2;
- batch 1-2: 16, as now.

It removes the combine (1.9-4.7 µs of GPU time plus its launch) and leaves a single wave. Expected
gain, if the DSMEM merge costs the same as at batch 1: about 2-4 µs per sliding shape at batch
16/32, on the order of +2-3% on the total. It is not the 2x, but it measures on the H200 whether the
single-kernel path scales with batch. It is the basis for points 1 and 2.

**H2 (later): overlapped `cl` prologue.** Batch 1 sliding is at 7.7 µs, and half of it is the
prologue. Issue `seq_lens` → page table at the start, before q and the new k/v, and load the pages
of the first block without waiting for q. Target: 7.7 → ~5 µs.
