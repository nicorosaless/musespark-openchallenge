# How the attention kernel works (`muse_attention_decode`)

This document explains, top to bottom, what the problem computes, how my solution distributes it
across the SMs of an H200, and why each piece has the shape it has. It assumes you know what a
warp, shared memory (smem) and a tensor-core MMA (*matrix multiply accumulate*) instruction are;
the rest is explained along the way. The numbers come from the judge's checks on the H200 unless
marked "bench" (the H100 of the university cluster).

## 1. The problem

One inference step of a model with grouped-query attention (GQA), per layer:

```
  qkv [tokens, (2·HQ + 2·HKV)·64]  bf16         tokens = batch · q_len
   ├─ q     HQ = 16 heads of 64     ─┐
   ├─ k     HKV = 2 heads of 64      │  GQA: each KV head is shared by
   ├─ v     HKV = 2 heads of 64      │  G = HQ / HKV = 8 q heads
   └─ gate  HQ = 16 heads of 64     ─┘

  K and V cache, E4M3 (FP8), paged with 1-token pages:
     page_table[b, t]  →  physical row of key t of request b
     row = HKV · 64 bytes = 128 B (both KV heads together), scale k_scale / v_scale
```

For each request b and each token of the step:

1. **New k:** RMSNorm and RoPE (rotation by position); it is quantized to E4M3 and written to the
   cache at `cache_loc`. The new v is quantized and written as is.
2. **q:** RMSNorm, RoPE and scaling.
3. **Attention:** `softmax(q·Kᵀ) · V` over keys `[lo, seq + j]`, causal within the tokens of the
   step. On *full* layers, `lo = 0`. On *sliding* layers, the window is the last 2048 keys.
4. **Output:** `RMSNorm(o) · sigmoid(gate)` in bf16.

Two regimes:

| | q_len | q rows per KV head | keys (full / sliding) |
|---|---|---|---|
| **decode** | 1 | 8 | 80,000 / 2,048 |
| **verify** (speculative decoding) | 8 | 64 | 80,000 / 2,048 |

With batch 1, 16 and 32 this gives six loads. Each load weighs 46 sliding layers and 16 full ones
(the real model), and the score is the **geometric mean** of the six speedups over SGLang (FA3 in
FP8).

**Precision.** The judge compares against an FP32 reference with a tolerance that allows q in E4M3,
because SGLang rounds it too. That is why I can do QKᵀ in FP8.

## 2. Split-KV and combine: how 80,000 keys are distributed

A decode request has 8 q rows per KV head: very little work per key. If a single CTA walked all
the keys, it would use one SM out of 132. So the keys are cut into pieces (*split-KV*):

```
  keys of request b, KV head h:          [ 0 ........................... 80,000 )
                                          │ split 0 │ split 1 │ ... │ split S−1 │
  CTA (s, h, b) ────────────────────────────▶ processes its chunk, block by block (64 keys)
                                               and leaves a partial:
                                                 o_s  = Σ p·v / Σ p        (normalized)
                                                 lse_s = log2 Σ 2^(score)  (base 2)

  combine (second kernel):    m = max_s lse_s
                              w_s = 2^(lse_s − m)
                              o  = Σ w_s · o_s / Σ w_s   →  RMSNorm · sigmoid(gate)  →  bf16
```

The normalized partial and its log-sum-exp are enough to merge in any order: it is the same trick
as FlashAttention's *online* softmax. The number of splits is chosen to fill one wave of CTAs:
`splits ≈ CTAs per SM · 132 / (batch · head pairs)`, capped at 128 (`MAX_SPLITS`). With batch 1 in
decode full that is 128 splits; with batch 32, about 12.

The combine is launched with **PDL** (*programmatic dependent launch*): it starts while kernel A is
finishing, loads the gate (which does not depend on A) and waits with `griddepcontrol.wait` before
reading the partials.

## 3. `full_fa2`: decode full (and sliding with batch ≤ 2)

A CUDA kernel ([`experiments/full_fa2.py`](experiments/full_fa2.py)), compiled in-process with NVRTC
and launched with `cuLaunchKernelEx`. One 128-thread CTA (4 warps) per (split, KV head pair,
request).

**Block-diagonal rows.** In decode, the 8 rows of each KV head are stacked into a 16-row tile (what
an m16 MMA needs): rows 0-7 are head 0 and rows 8-15 head 1. Each row only attends over the
dimensions of its own head. This way the whole 128 B cache row (both heads) is read, at the cost of
wasting half of the MMA.

```
           head 0 dims (64)   head 1 dims (64)
         ┌──────────────────┬──────────────────┐
 rows    │  q of head 0     │      zeros       │  ← 8 rows
 0-15    ├──────────────────┼──────────────────┤
         │      zeros       │  q of head 1     │  ← 8 rows
         └──────────────────┴──────────────────┘
```

**The copy pipeline (`cp.async`).** The gather is random: with 1-token pages, each key sits in a
different physical row. For each block of 64 keys the kernel must first read its 64 page entries
and then the 64 rows. That is two dependent DRAM hops, plus the `seq_lens` hop at the start.
`full_fa2` pipelines this by hand:

```
  ring of STAGES stages in smem (raw K and V, E4M3)          page entries
  ┌────────┬────────┬────────┐                               2·STAGES − 1 blocks ahead
  │ block  │ block  │ block  │ ◀── cp.async (16 B/thread)    ┌──┬──┬──┬──┬──┐
  │   j    │  j+1   │  j+2   │                               │  │  │  │  │  │ ◀── cp.async (4 B)
  └────────┴────────┴────────┘                               └──┴──┴──┴──┴──┘
      ▲ being computed                                       the row of each key
      │                                                      is already in smem
  iteration j:  cp.async.wait_group(STAGES − 2) → __syncthreads → issue pages of j + 2S − 1
                and copies of j + S − 1 → compute block j
```

`cp.async` copies from global memory to smem without going through registers and without blocking
the thread. The groups (`commit_group` / `wait_group`) tell when each block has arrived.

**The per-block computation** (FlashAttention-2 in registers):
1. **S = QKᵀ** in E4M3 (`mma.sync m16n8k32`). The B fragments of K are read straight from the
   stage, with the dimensions permuted the same way in Q.
2. **Online softmax:**
   - row maximum with `shfl`;
   - p = 2^(s − m);
   - rescale O by 2^(m_old − m_new).
3. **O += P·V** in FP16 (`mma.sync m16n8k16`). V is converted from E4M3 to FP16 in registers.

In decode, the 4 warps share the 64 keys of each block (16 each) and at the end merge their 4
states through smem.

## 4. `verify_ws`: verify full with specialized warps

In verify there are 64 q rows per KV head (8 heads × 8 tokens): one whole m64 MMA, which is what
`wgmma` does. `wgmma` is Hopper's asynchronous MMA, executed by a *warpgroup* (4 warps, 128 threads)
with operands in smem. `verify_ws` separates copying from computing:

```
  256-thread CTA (4 per SM)
  ┌──────────────────────────────┐        ┌──────────────────────────────────────┐
  │ producer warpgroup           │        │ consumer warpgroup                   │
  │ setmaxnreg.dec → 24 registers│        │ setmaxnreg.inc → 104 registers       │
  │ (1 warp copies, 3 exit)      │        │                                      │
  │                              │ full[s]│                                      │
  │ wait empty[s]     ───────────┼───────▶│ wait full[s]                         │
  │ pages (PSMEM ring)           │        │ S = Q·Kᵀ   wgmma m64n64k32 e4m3      │
  │ cp.async K, V → stage s      │        │ V E4M3 → FP16 in smem                │
  │ cp.async.mbarrier.arrive     │        │ online softmax (64 × 64)             │
  │   .noinc → full[s]           │◀───────┼ O += P·V   mma.sync                  │
  │                              │empty[s]│ arrive on empty[s]                   │
  └──────────────────────────────┘        └──────────────────────────────────────┘
```

- **mbarriers.** These are barriers in smem with an arrival counter and a phase. `full[s]`
  completes when the copies of stage s have arrived (each producer lane "arrives" when its
  `cp.async` copies land). `empty[s]` completes when the 4 consumer warps are done with the stage.
  Nobody calls `__syncthreads`: producer and consumer only wait on the stage they share.
- **`setmaxnreg`.** It moves registers from the producer (which barely uses them) to the consumer.
  The average is 64 per thread, so 4 CTAs fit per SM: 16 compute warps.
- **`PSMEM`.** The page entries go through `cp.async` into a 5-slot ring in smem, 4 blocks ahead.
  Before this, the producer stalled on every page-table load.
- **`fence.proxy.async`.** `wgmma` reads smem through the *async proxy*, and `cp.async` writes
  through the generic proxy. This fence is needed between the two.

## 5. `cl`: a single kernel with clusters and DSMEM (sliding with batch ≤ 2)

On the sliding layers with batch 1 everything is latency: 2048 keys are 0.5 MB, nothing for the
DRAM. The cost is two launches (kernel A + combine) and the serial hops. `cl` does it in one kernel:
the 16 CTAs of the splits form a Hopper **cluster**, which is scheduled together on one GPC and can
write into the other CTAs' smem (DSMEM, *distributed shared memory*).

```
  cluster of 16 CTAs = 16 splits of 128 keys of the same request
  ┌────┐┌────┐┌────┐      ┌────┐
  │ r0 ││ r1 ││ r2 │ .... │r15 │   each CTA: its normalized partial (16 rows × 64) + lse
  └─┬──┘└─┬──┘└─┬──┘      └─┬──┘
    │ remote write (map_shared_rank) into the smem of each row's "owner": rank ρ % 16
    ▼      ▼      ▼           ▼
  barrier.cluster (arrive.release + wait.acquire)
    │
  each owner merges its rows from its own smem → RMSNorm · sigmoid(gate) → bf16 to global
```

- `barrier.cluster.arrive` at the start and `wait` before the first remote write: this way no CTA
  writes into the smem of another one that has not started yet.
- **`NEWSMEM`:** the split of block 0 writes the new k/v to the cache and also into its own stage.
  This saves reading it back from global memory.
- **`EARLYQ`:** the loads of q and of the new k/v are issued before `seq_lens` is known.
- **With `KVP = 1`** (since `f31f6572`, also in the mark `28dd5b80`): two clusters per request, one
  per KV head. Each CTA does 8 rows × 64 dimensions instead of 16 × 128. On the bench, decode 1
  sliding goes from 10.2 to 9.0 µs.

## 6. What did not win on the H200

The full table, with the check of each idea and its µs, is in [`what-didnt-win.md`](what-didnt-win.md).
In short:
- **verify full**: twelve variants of the consumer and of the work split (`KVP = 2`, `PP`, `PVA`,
  `PVA = 2`, `OVL`, BN 32 / 128, `EXPH`, `CLW`…) ended up between 2% and 45% above 166-170 /
  317-322 µs;
- **decode full**: more stages, more CTAs or half a row per CTA, a tie or worse;
- **sliding**: one kernel with the cluster sized to one wave at batch 16/32, worse; `KVEARLY` and
  `EPIW` alone, within the noise (they ended up in the mark together with two other ideas).

**The bench lesson.** The cluster's H100 has ~1.5 TB/s (HBM2e) and the H200 4.8 TB/s. On the bench,
verify full is limited by DRAM efficiency with random rows, and anything that reads whole 128 B rows
wins. On the H200 something else rules: **how many CTAs there are per SM and how long the
dependency chain of each block is**. Three times, the bench said "−30%" and the H200 said "+18%".

```
  rule of thumb
  ┌───────────────────────────────────┬────────────────────────────────┐
  │ change                            │ decided on the bench?          │
  ├───────────────────────────────────┼────────────────────────────────┤
  │ latency (prologue, combine,       │ yes (decode 1 sliding, combine)│
  │ launches)                         │                                │
  │ CTAs/SM, MMA, row split,          │ no: it needs a check on        │
  │ gather shape                      │ the H200                       │
  └───────────────────────────────────┴────────────────────────────────┘
```

One more detail also misled me: ptxas **serializes every `wgmma`** in the kernel (warning C7514 or
C7515) if any ordinary instruction reads or defines an accumulator while a `wgmma` is in flight,
even on a path that never runs at the same time. The `PVA` of 26 Oct was measured serialized. Since
then, every variant is assembled with the judge's ptxas and checked for those warnings before it is
measured.

## 7. The gap: where I stand

H200 floors: bytes at 4.8 TB/s; compute in FP8 at 1979 TFLOP/s. Details in
[`gap-to-top40.md`](gap-to-top40.md).

| shape | my µs | bytes floor | × floor |
|---|---|---|---|
| decode 16 / 32 full | 110 / 201 | 69 / 139 | 1.6 / 1.5 |
| verify 16 / 32 full | 170 / 322 | 70 / 139 | 2.4 / 2.3 |
| decode 1 / verify 1 full | 20 / 21 | 4.3 / 4.4 | 4.6 / 4.8 |
| sliding (all) | 9-24 | 0.1-4 | 5-84 |

```
  × floor   1      2      5     10     50    100
            │──────┼──────┼─────┼──────┼──────┤
  decode full █▌
  verify full ██▍
  batch 1 full     ████▋
  sliding          █████─────────────────────── (up to 84)
```

The large-batch full shapes are already close to their floor. Rank 40 on the board (2.52x) cannot
win there: it would have to run at 97-106% of peak HBM bandwidth in decode full. It wins on the
**sliding and batch-1 shapes**, which are latency: launches, three serial DRAM hops (`seq_lens` →
page → row) with a cold L2 (every judge replay starts that way) and the merge of the splits. With
the sliding layers at ~4 µs, my total would go from 1.69 to 2.18.

That is why the last two marks attack latency. `f31f6572` (1.760x): `cl` with one head per CTA and
the combine in a single pass of 128 splits. On the H200, decode 1 sliding goes from 9.06 to 8.09
µs, decode 1 full from 20.05 to 16.31 and verify 1 full from 20.90 to 18.33. `28dd5b80` (1.801x, the
mark): verify sliding also goes through `verify_ws`, which brings verify 1 / 16 / 32 sliding down
from 10.0 / 17.4 / 23.9 to 9.0 / 15.3 / 20.5 µs.
