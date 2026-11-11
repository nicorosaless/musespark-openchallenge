# MoE with NVFP4 weights on an H200: the concepts

This note explains, for a reader who knows the basics of CUDA, how the `muse_moe_w4a16`
kernel works and why it is built this way. The numbers come from the judge
checks in [`evidence/runs/muse_moe_w4a16/`](../../evidence/runs/muse_moe_w4a16/) and the bench runs
in [`evidence/bench/muse_moe_w4a16/`](../../evidence/bench/muse_moe_w4a16/).

## 1. The problem

An MoE (*mixture of experts*) layer has 256 experts, and each token goes to 8 of them (top-8). Each
GPU holds 32 experts (EP8). An expert is an MLP with SwiGLU:

```
             x [tokens, 4096] (BF16)
                   │
       ┌───────────┴───────────┐
   x · Wgateᵀ               x · Wupᵀ           GEMM1: W13 = [gate; up], 8192 × 4096
       │                       │
   silu(gate)      ×           up               inter [tokens, 4096]
                   │
            inter · W2ᵀ                         GEMM2: W2, 4096 × 4096
                   │
       RMSNorm · (1 + w) · router weight        then the token's 8 experts are summed
```

The judge times *decode* shapes (1-32 tokens per step) and *verify* shapes (128 and 256 tokens,
which the layer's breakdown calls "verify 16/32"). With so few tokens, each expert sees 1 to ~10
tokens. The GEMMs are almost **GEMVs**: a lot of weights are read to do very little arithmetic with
each one.

Per GPU there are 32 experts × (8192 + 4096) × 4096 weights ≈ 1.6 G weights ≈ 0.8 GB at 4 bits. At
4.8 TB/s (H200), reading them once costs ~170 µs if every expert is touched. That is the order of
the memory floor (`speed_of_light_ns` in `reference.py`).

## 2. NVFP4 and how to dequantize it with bit operations

### 2.1 The format

Each weight is a 4-bit **E2M1**: 1 sign bit, 2 exponent bits and 1 mantissa bit. It can only take
the values ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. To make that useful, each block of 16 consecutive weights
carries an **E4M3 scale** (FP8), and each tensor carries a **global FP32 scale**:

```
   real weight = e2m1 × block_scale(E4M3) × global_scale(FP32)

   weight byte:    [ s e1 e0 m | s e1 e0 m ]      two codes per byte
   per 16 weights: 1 E4M3 scale byte
```

The challenge rules require the weights to stay in NVFP4 (they may be **reordered** losslessly in
`prepare()`, but not converted to BF16) and the activations to be BF16. The H200 tensor cores do not
multiply FP4 × BF16, so the kernel has to **dequantize**: convert each code to BF16 in registers
right before the multiply. With so few tokens, that decode is the dominant work, and the kernel ends
up limited by the decode's integer instructions, not by memory or by the tensor cores.

### 2.2 The subnormal trick: decoding without tables or conversions

A BF16 is `s | eeeeeeee | mmmmmmm` (1 + 8 + 7 bits). If the bits `e1 e0 m` of the E2M1 code are
written into positions 8..6 of the BF16, with the upper part of the BF16 exponent at zero, the
resulting number is a BF16 **subnormal** worth exactly `e2m1 × 2^-126`:

```
   E2M1 code:              s  e1 e0  m
                           │   └─┬──┘
                           ▼     ▼
   BF16:          bit 15 = s,  bits 8..6 = e1 e0 m,  rest 0
                  = 0 0000000 1 1 0 0000   (example: e1e0m = 110 → 4 × 2^-126)
```

This works because E2M1 with e = 0 is also "subnormal" (0 or 0.5): BF16 arithmetic interprets it the
same way. Then **one** `mul.bf16x2` (two values at a time) by `scale × 2^56` leaves
`e2m1 × scale × 2^-70`, exactly: 2 + 4 significant bits fit in the 8 of BF16. The missing `2^70`
factor, together with the global scale, is applied once in the epilogue, on the FP32 accumulator.

The scale is also prepared with bit operations. The BF16 of `scale × 2^56` for a normal E4M3 is
`0x5800 | (b & 0x7F) << 4 | sign`. In v9, `prepare()` recodes each scale byte (a bijection of its
8 bits) so that the BF16 pair `(s, s)` comes out of **one PRMT** (replicates a byte into all four)
and **one LOP3** (`(W & 0x07F007F0) | 0x58005800`). If a tensor had subnormal scales, `prepare()`
detects it and that GEMM takes an exact path with two FP32 multiplications.

### 2.3 The bit reordering (v8)

A 32-bit word holds 8 codes. The mma needs 4 registers of 2 BF16 each (the "fragment"). With the
nibbles in their natural order, placing each code at positions 15 and 8..6 of its 16-bit half costs
many shifts and masks. But `prepare()` can put **each bit wherever I want**, as long as the 4 bits
of a code are not lost. [`tools/search_bit_layout.py`](tools/search_bit_layout.py) searches for
the placement of the 4 codes of each half that minimizes the operation count:

```
   word q (32 bits) = two 16-bit halves; each half, 4 codes:

   bit:     15  14  13  12 | 11 10  9 |  8  7  6 |  5  4  3 |  2  1  0
   (v23)    s0  s2  s3  s1 |    m3    |    m0    |    m1    |    m2

   target of each register j (per half): sign at bit 15, magnitude at 8..6  → mask 0x81C081C0

   register 0:  q & 0x81C081C0               already in place: 1 LOP3
   register 1:  (q << 3) & 0x81C081C0        sign (12) and magnitude (3..5) move up 3: 1 shift + 1 LOP3
   register 2:  sign (14) moves up 1, magnitude (0..2) moves up 6: two different shifts (see 2.4)
   register 3:  sign (13) moves up 2, magnitude (9..11) moves down 3
```

v8 cut the decode from 15 to 11 operations per word (−16% instructions) and the speedup rose 16%
(1.378x → 1.601x). It was the proof that the kernel was bound by instructions.

### 2.4 The multiplication trick (v23)

Some registers need two different shifts (the sign goes to one place and the magnitude to another).
An integer multiplication does **two shifts at once**: `x × (2^a + 2^b) = (x << a) + (x << b)`. If
the two copies do not overlap (no carries), the sum is an OR:

```
   register 2:  ((q & 0x40074007) × 66) & 0x81C081C0        66 = 2^1 + 2^6

   (one 16-bit half; bit 15 on the left)
   q & 0x4007     0 s 0 0 0 0 0 0 0 0 0 0 0 m m m       sign at 14, magnitude at 2..0
   × 2            s 0 0 0 0 0 0 0 0 0 0 0 m m m 0       copy 1: the sign is already at 15
   × 64           0 0 0 0 0 0 0 m m m 0 0 0 0 0 0       copy 2: the magnitude is already at 8..6
                  (the sign of copy 2 leaves the top of the half, to bit 20: the mask removes it)
   sum = OR       s 0 0 0 0 0 0 m m m 0 0 0 m m m       no overlap: no carries
   & 0x81C0       s 0 0 0 0 0 0 m m m 0 0 0 0 0 0       the BF16 register, ready
```

One IMAD replaces two shifts and an OR. [`tools/search_imad_layout.py`](tools/search_imad_layout.py)
enumerates every bit order that allows it. Only one reaches the minimum, 10 integer operations per
word: `CODE_SIGN_BIT = (15, 12, 14, 13)`, `CODE_MAG_LOW = (6, 3, 0, 9)`. v23 reached 1.768x, and
v24, which also simplifies addressing, **1.811x** (submit `fc6aaf56`). The final version, v55
(section 7), stands at **1.8585x** (submit `db79106e`, rank 538 of 1203).

## 3. The GEMM without shared memory: `mma.sync` and swap AB

### 3.1 Swap AB

`mma.sync.m16n8k16` multiplies A (16 × 16) by B (16 × 8). With few tokens, if the tokens were the
M side (16 rows), a block of 3 tokens would waste 13 of the 16 rows. So the roles are swapped
(*swap AB*): the **weights are A** (16 output rows) and the **tokens are B** (8 columns):

```
       outputᵀ    =     W (weights)     ·    xᵀ (tokens)
   [16 rows × 8 tok]    [16 × 16 of K]       [16 of K × 8 tok]
                            ▲                      ▲
                     decoded in              BF16 activations,
                     registers (A)           straight from memory (B)
```

### 3.2 One warp, one item, no shared memory

Each warp works alone (persistent: it keeps taking items until none are left). An item is (block of
≤ 16 tokens of one expert, 16 output rows, K range). For each 64-wide chunk of K:

```
   lane l (32 lanes):  1 load of 16 B = 32 codes of its weights      ──┐
                       its 16 activations per token (contiguous)       │  all in registers
                       decode → 4 BF16 registers × 4 k16 steps         │
                       4 mma.sync (or 8 with two 8-token tiles)      ──┘
```

`prepare()` reorders K inside each chunk (`real k = 16c + 4kk + 2h + b`) so that the fragment the
mma expects from each lane is **a single 16-byte load**, and so that that lane's 16 weights fall in
the same scale block. Since the sum over K does not depend on the order, it is enough for
activations and weights to use the same permutation.

No shared memory or barriers are needed: all the parallelism comes from independent warps (4 CTAs ×
4 warps per SM). Memory latency is hidden by many warps in flight and by 8 KB bursts per warp.

## 4. Split-K

With 1-2 tokens there are very few items (blocks × columns) to fill 132 SMs × 16 warps. **Split-K**
splits the K sum across several items, and each one writes an FP32 partial sum:

```
   K = 4096:   [ item 0: K 0..1023 ][ item 1: K 1024..2047 ][ ... ]
                      │                     │
                      ▼                     ▼
                FP32 partial           FP32 partial        → summed in the next step
                                                             (_swiglu_reduce or _combine)
```

Each item takes 1/4 of K at bs ≤ 2 and 1/2 at bs ≤ 64. Besides filling the GPU, shorter items
shorten the tail of the last round of the persistent grid: with 1.5 rounds of items, half the GPU
does the second half alone. The partials are not summed in a separate kernel: the sum lives inside
steps that already exist (SwiGLU for GEMM1 and the final combine for GEMM2).

## 5. wgmma: what was tried and why the H200 judged it differently from the bench

### 5.1 What wgmma is

On Hopper (sm_90a), `wgmma.mma_async` is a **warpgroup** MMA (4 warps, 128 threads): M = 64, N from
8 to 256, asynchronous (issue it, then wait with `wait_group`). B must be in shared memory; A can be
in registers with the same fragment as `mma.sync`, so the decode works unchanged:

```
   CTA (128 threads = 1 warpgroup)
   ┌──────────────────────────────────────┐
   │ warp 0: decode rows  0-15   (regs A) │
   │ warp 1: decode rows 16-31   (regs A) │──► wgmma m64nNk16 ──► accumulators
   │ warp 2: decode rows 32-47   (regs A) │        ▲
   │ warp 3: decode rows 48-63   (regs A) │        │
   └──────────────────────────────────────┘   B: N tokens in shared
                                                (copied once per CTA)
```

The idea was for 32-token blocks to use a large N with one decode per weight, instead of one decode
per 16-token block.

### 5.2 What the bench (H100) said and what the H200 said

| | H100 (bench) | H200 (judge, layer profile) |
|---|---|---|
| v41 (both GEMMs with wgmma), verify 32 | −18% | gemm1 −2%, gemm2 +4% |
| v41, verify 16 | −3.7% | gemm1 +13%, gemm2 +27% |

The bench is an H100 with the same torch version; the H200 is the judge's machine. Both GPUs have
the same SM (Hopper), but the H200 has much more memory, and wider (4.8 TB/s against the ~1.6 TB/s measured on the bench's 64 GB H100):

```
   H100:  DRAM at ~90%  →  memory sets the time; extra work hides behind it
   H200:  DRAM to spare  →  the ALU and the stalls set the time; extra work is paid for
```

On the H100, wgmma won because it decoded each weight fewer times, and with DRAM saturated that was
the only thing that showed. On the H200, costs that the H100 hid appear:

- **Granularity.** A wgmma item is a whole CTA (4 warps on one block), 4 times coarser than the
  per-warp item of v24. The tail of the last round of the persistent grid costs the same in
  absolute terms in gemm1 and gemm2, and that is exactly what the profile shows: +16-19 µs in both.
  gemm2, with half the items, loses more in relative terms.
- **Wasted tensor work** with N = 32 on blocks of 4-6 tokens. It was the first hypothesis, but it
  does not fit: gemm1 issues twice the wgmma of gemm2 and does not lose twice as much.

The batch of 28 Oct measured each answer separately on the H200 (total speedup; v24 is at 1.812,
with bs 128 2.018 and bs 256 1.980):

| variant | total | bs 128 | bs 256 |
|---|---|---|---|
| v41, N = 32 always | 1.733 | 1.641 | 1.839 |
| v45, N by block | 1.622 | 1.412 | 1.429 |
| v46, `mma.sync` for blocks ≤ 16 | 1.684 | 1.586 | 1.591 |
| v47, split-K 2 | 1.726 | 1.647 | 1.806 |
| v48, dynamic per-CTA item queue | **1.757** | **1.710** | **1.917** |

None reaches v24. The only one that beats v41 in both shapes is the dynamic queue, the granularity
lead. That idea, applied to the per-warp items of v24, is what won in the end (section 7).

**Lesson:** the bench is good for correctness and for what depends on memory, but it does not
predict decode, ALU or MMA changes on the H200. Those are measured on the H200.

## 6. Warp specialization

### 6.1 The problem it attacks

In v24 each warp does everything in sequence: load weights, decode, multiply. While it waits on
memory or on the mma, nothing else in that warp moves. The issue floor of verify 32 is ~66 µs of
instructions, and the two GEMMs take ~290 µs. Issue runs at 25-30%, so the limit is the stalls
inside each warp.

### 6.2 The idea

The roles inside the CTA are separated. A **producer** warpgroup only loads weights and decodes them
to BF16 in shared memory. A **consumer** warpgroup only issues wgmma, with A (decoded weights) and B
(tokens) in shared memory. Between the two there is a **ring of stages** synchronized with
*mbarriers*:

```
              producer (4 warps)                           consumer (4 warps)
   ┌──────────────────────────────────┐           ┌──────────────────────────────────┐
   │ cp.async raw weights → ring      │           │ cp.async block tokens → B        │
   │ wait empty[s]                    │           │ wait full[s]                     │
   │ decode → stmatrix → A[s] (BF16)  │──full[s]─►│ wgmma(A[s], B[s]) N = 16/32/64   │
   │ arrive full[s]                   │◄─empty[s]─│ arrive empty[s]                  │
   └──────────────────────────────────┘           └──────────────────────────────────┘

   shared, 3 stages:  [A0 16 KB | B0 8 KB][A1 | B1][A2 | B2]  barriers  [raw weight ring 32 KB]
```

- An **mbarrier** is a barrier in shared memory with an arrival counter and a phase bit. `full[s]`
  says "stage s has weights ready" and `empty[s]` says "the consumer is done with stage s". Each
  side waits on the parity of its own lap, so nobody has to reset anything.
- **setmaxnreg** splits registers asymmetrically: the producer drops to 104 and the consumer rises
  to 152 (for the accumulators). The numbers must match the launch's register count exactly, or the
  kernel hangs. It happened: a micro-test hung for 20 minutes on the bench.
- Each weight is decoded **once per expert and call** for all its tokens (blocks of up to 64), and
  load, decode and MMA overlap across stages.

### 6.3 Where it stands

| milestone | variant | status |
|---|---|---|
| 1. gemm1 of one expert (micro-test) | [`diagnostics/ws_micro.py`](diagnostics/ws_micro.py) | correct on H100 against float64; stalls at T = 128: producer 6.5%, consumer 10.5% |
| 2. persistent grid over experts (gemm1) | v51, v52 (16-entry table for gemm2), v53 (raw weight ring) | correct on the H100 (5/5); on the H200, 1.497-1.515 |
| 3. gemm2 specialized too | v54 | correct on the H100 (5/5); on the H200, 1.418 |

### 6.4 What the H200 said

On the H100 it tied. On the H200 the two specialized GEMMs take **twice as long** as v24:

| µs on the H200 | gemm1 bs 128 | gemm1 bs 256 | gemm2 bs 128 | gemm2 bs 256 |
|---|---|---|---|---|
| v24 (independent warps) | 155 | 197 | 74 | 96 |
| v53 (specialized gemm1) | 334 | 382 | 76 | 95 |
| v54 (both specialized) | 343 | 459 | 177 | 234 |

The reason is the warp count. In v24, 16 warps per SM decode (4 CTAs × 4 warps). With
specialization, 2 CTAs × 4 producer warps = **8**. The decode is limited by issue rate and latency
inside each warp, so with half the warps decoding it takes almost twice as long. Overlapping with
the MMA does not make up for it, because the MMA was never the bottleneck: at verify 32 the
arithmetic is ~2.6·10^10 FLOP, ~26 µs of tensor core at 990 TFLOP/s, against ~290 µs of GEMM. To
pay off it would need as many producer warps as v24 has, and they do not fit in the registers the
consumer leaves.

## 7. The dynamic item queue (what won in the end)

In v24 the work split is static: warp `gw` does items `gw`, `gw + warps`, `gw + 2·warps`… But items
do not cost the same: a block of more than 8 tokens runs two MMA tiles (NT = 2), and one of up to 8
tokens runs only one. In the last round, some warps still have long work while others are done.

With a **dynamic queue**, each warp requests its next item from a global counter with `atomicAdd`:

```
   static:     warp 0: ■■■■ ■■ ■■■■ ■■■■         ← finishes late
               warp 1: ■■ ■■ ■■ ■■   (idle)
   dynamic:    warp 0: ■■■■ ■■ ■■■■ ■■
               warp 1: ■■ ■■ ■■ ■■ ■■■■ ■■        ← both finish at almost the same time

   loop:   item = gw                                (the first one without an atomic: does not delay bs 1)
           repeat:   tile_loop(item)
                     lane 0: next = atomicAdd(&counter, 1)    ← its L2 round trip overlaps the epilogue
                     epilogue(item)
                     item = warps + shfl(next)                 ← each value is handed out once
           until item >= items
```

The counter lives in `nblk[2..3]` (one per GEMM). The route step (`moe_route` or `_route_prep`)
resets it to zero on every call, inside the kernel graph itself: never from Python, and also under
the judge's CUDA graph replay.

| H200 | total | bs 128 | bs 256 |
|---|---|---|---|
| v24 | 1.812 [1.785, 1.839] | 2.018 | 1.980 |
| **v55** = single file (v24 + PDL chain up to bs 16) + queue | **1.863** (two checks: 1.8627 and 1.8637; submit `db79106e`: 1.8585) | 2.095 | 2.031 |
| v56 = v55 + reserving the next item to fetch its table ahead | 1.831 | 1.989 | 1.973 |

At verify 16/32, where there is no PDL, the queue alone gives +3.8% and +2.6%. v56 loses:
reserving an item ahead leaves each warp with work set aside, and that brings back part of the
imbalance the queue removed.

The full list of what didn't win, with its numbers, is in [`what-didnt-win.md`](what-didnt-win.md).
