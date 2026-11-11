# The Muse Spark decoder layer on an H200: how it is built and why

The decode_layer problem of the Open Kernel Challenge, 27 Oct 2026, revised at the close (30 Oct). It
assumes basic GPU programming: threads, warps, blocks (CTAs), SMs, shared memory, L2 and DRAM, and
what memory-bound and compute-bound kernels are. Everything else is explained here.

## 1. What the layer computes

It is one Muse Spark decoder layer as one of the 8 GPUs of an expert-parallel node runs it. Each
step processes T tokens: T = batch × q_len. In decode, q_len = 1 and T goes from 1 to 32. In
verify, each request brings an 8-token draft, so T goes from 8 to 256.

Inputs:
- `x` [T, 8192] in BF16, the attention input (already normalized by the previous layer);
- `s` [T, 8192] in FP32, the residual stream;
- the FP8 KV cache, with up to ~80,000 tokens of context per request.

Outputs: the input of the next layer, the new residual stream, and the 8 experts chosen per token.

Step by step (TASK.md):

```
q, k, v, g = split(x @ qkv_gate.T)        qkv projection: [T, 8192] x [8192, 2304]
                                           q 16 heads x 64, k and v 2 heads x 64, g the gate (1024)
QK-norm and RoPE; new k and v to the FP8 cache
a = (rmsnorm(attention(q)) * sigmoid(g)) @ o.T   attention + o projection: [T, 1024] x [1024, 8192]
s = α1·s + β1·rmsnorm(a);  h = rmsnorm(s)·(1 + w)        post-attention boundary
experts = top8(sigmoid(0.177 · h @ router.T) + bias)      router: [T, 8192] x [8192, 256], FP32 weights
y = local experts(rmsnorm(h @ pooler.T)·(1 + w))          pooler: 8192 -> 4096; MoE with NVFP4 weights
f = y @ unpooler.T                                         unpooler: 4096 -> 8192
s = α2·s + β2·rmsnorm(f·(1 + w))                           post-FFN boundary
return rmsnorm(s)·(1 + w), s, experts
```

The communication between GPUs (all-reduce, all-gather) is removed: it counts as the identity. That
is why, with 8 tokens or fewer, the unpooler only computes the 1,024 columns that belong to this
GPU and the rest is zero.

Two layer types:
- **sliding**: the attention only looks at the last 2,048 keys. There are 46 per step.
- **full**: it looks at the whole context, ~80,000 keys. There are 16 per step.

The MoE has 256 experts. Each token picks 8, and 32 of them live on this GPU (local ids 0-31). A
token uses ~1 local expert on average, but the distribution is skewed: a few experts take most of
the tokens.

## 2. What the judge measures

- Six loads: decode 1, 16 and 32, and verify 1, 16 and 32 (the number is the batch).
- Each load adds one sliding layer × 46 and one full layer × 16, the weight each has in a real step.
- The score is the geometric mean, over the six loads, of SGLang's time divided by mine. Each load
  counts the same: a 10% gain in decode 1 is worth as much as in verify 32, even though one takes
  100 µs and the other 700.
- Each shape is captured as a CUDA graph and timed by replaying it, with a cold L2, alternating with
  SGLang's kernel on the same inputs. The ratio for each shape is the median of those pairs.
- Before timing, 13 correctness tests run, some harder than the scored shapes (contexts of 10 keys,
  outliers, batch 9...). With CUDA graphs nothing can be read back on the host and nothing can
  branch on the data.

My mark: **1.6754x** (submit `b76b7763`, v45): on average 1.68 times faster than SGLang. The
overall ranking has the same order as this problem. On the 27th at 18:45 it was rank 280 in
decode_layer (281 overall); on 30 Oct, with a much fuller board, rank 511 in decode_layer (515
overall). By then rank 40 required 2.252x.

The final version in the repo is v54 (`3b35898`, sha256 `7aa4a1a1…`; check `54914d08`, 1.6684x). It
carries the new attention (`f31f6572`), the MoE PDL chain only up to 16 tokens, and the MoE kernel's
single experts file (v55, with a dynamic work-item queue). Its predecessors measured on the 28th and
30th (v48, v52, v53: 1.666-1.679) and v54 itself stay within the noise of v45 and do not beat the
upper bound of the mark's 95% CI (1.7005), so none was submitted.

## 3. What pieces it is made of

The layer is built from four pieces. Three come from the standalone problems, and the layer's own
code composes them:

| piece | source | what it contributes |
|---|---|---|
| attention (kernel A + combine) | attention | Triton and CUDA (NVRTC); split-KV; on sliding with batch ≤ 2, a cluster of CTAs (one head per CTA since `f31f6572`) that merges the splits through distributed shared memory |
| dense projections | bf16 | c1: the BF16 weights losslessly compressed to 12 bits (exact correction tables), and split-K for when there are few tokens |
| experts (MoE) | MoE, and bf16 for ≤32 tokens | CUDA GEMMs that decompress NVFP4 in registers with `mma.sync` (MoE v24, with v55's dynamic work-item queue); bf16's PDL chain up to 16 tokens |
| routing, norms, boundaries, dispatch, fusions | this kernel | `route_pre_expert`, the post-FFN boundary, the fused finish, the streams and the layer's PDL chain |

Each piece comes in through a script that copies the section from the other kernel and adapts its
`kernel_fn` into a function of the layer (`patch_attn.py`, `patch_moe.py`, `integrate_moe.sh`, in
[`tools/`](tools/)). Then come the bench (the 13 tests on an H100) and a check on the judge's H200.
A version goes to the board only if the check beats the upper bound of the current mark's 95% CI.

## 4. The layer's flow

With 1 token (decode 1), the layer launches about 17 kernels on the main stream and 2 on the side
stream:

```
 main stream                                              side stream
 ───────────                                              ───────────
 qkv: c1 GEMV stream-K ─▶ c1 finish
          │
          ▼
 attention: kernel A (cluster) [+ combine on full]
          │
          ▼
 o: c1 GEMV ─▶ c1 finish
          │
          ▼
 post-attention boundary (SGLang) ──── h = ffn_input ─────▶ (waits for main)
          │                                                 router: split-K reds
          ▼                                                 router: finish ─▶ logits
 pooler: c1 GEMV ─▶ c1 finish                                   │
          │                                                     │
          ▼  ◀───────────────── waits for side ─────────────────┘
 route_pre_expert: top-8, weights, local ids, pre-expert norm
          │
          ▼
 MoE: route prep ─▶ gemm1 ─▶ swiglu reduce ─▶ gemm2 ─▶ combine     (PDL chain up to 16 tokens)
          │
          ▼
 unpooler (1024-row slice): c1 GEMV
          │
          ▼
 post-FFN boundary with the unpooler finish inside ─▶ outputs
```

With more tokens the projection paths change:

| tokens | qkv | o | pooler | unpooler |
|---|---|---|---|---|
| 1 | c1 GEMV stream-K | c1 GEMV | c1 GEMV | c1 GEMV of the slice |
| 2-8 | split-K | cuBLAS | c1 with tensor cores | split-K of the slice |
| 9-16 | split-K | cuBLAS | c1 with tensor cores | c1 with tensor cores |
| 17-32 | split-K | cuBLAS | split-K | cuBLAS (NN) |
| 128-256 | cuBLAS | cuBLAS | cuBLAS (NN) | cuBLAS (NN) |

The dispatch depends only on the shape (T, the layer type), never on the data. So each shape has a
fixed graph.

## 5. The two streams

The router and the pooler read the same input (`h`) and do not depend on each other. The pooler is
large (67 MB of BF16 weights) and the router small (8 MB: its FP32 weights go as a BF16 hi + lo
pair). So the router runs on a second CUDA stream, in parallel with the pooler:

```
 time ───────────────────────────────────────────────────────▶
 main      │ boundary │ pooler GEMV ................ │ finish │ route_pre_expert │ MoE ...
 side      │          │ router reds │ finish │                ▲
                      ▲ fork (wait_stream)               join ┘ (wait_stream)
```

In a CUDA graph, `wait_stream` becomes an ordinary edge between nodes: the next kernel waits for the
other one to finish completely. Each stream has its own workspaces (section 7) so they do not mix
partial sums.

v47 tried merging pooler and router into a single GEMM with FP32 output. It loses on the H200:
the joint GEMM lengthens the main stream, and the router was already hidden behind the pooler.

## 6. The PDL chain

Between two consecutive kernels on the same stream there is a gap. The second one does not start
until the first has finished completely and its memory is visible: launching, distributing CTAs and
starting up costs ~1-3 µs. The layer has ~17 kernels in series, and with few tokens many of them
take 2-10 µs, so that gap matters.

PDL (programmatic dependent launch, Hopper) lets the next kernel start earlier. It takes two PTX
instructions:

- `griddepcontrol.launch_dependents` in the primary kernel: "the next one may launch now". Once all
  the primary's CTAs have executed it (or finished), the dependent can start occupying free SMs.
- `griddepcontrol.wait` in the dependent: it waits until the primary has **finished completely** and
  its memory is visible.

The dependent is launched with an attribute (`launch_pdl=True` in Triton;
`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` with `cuLaunchKernelEx`). Without that
attribute, the two instructions do nothing.

```
 no PDL:    primary  ████████████│ gap │ dependent ░░░░░▓▓▓▓▓▓▓▓
 with PDL:  primary  ████████████│
                     dependent    ░░░░░⏸▓▓▓▓▓▓▓▓
                                  │    └ griddepcontrol.wait: here the primary has already finished
                                  └ starts as soon as the primary fires launch_dependents
            ░ = prologue that does not depend on the primary (weights, constants)   ▓ = what reads the primary's output
```

**The rule.** Before its `wait`, a dependent may only read things its primary does not write:
weights, constants, or data written by a kernel that finished before the primary started. Reading
the primary's output before the `wait` is a race: sometimes the data is there and sometimes not.
Every version is checked against this rule down to the PTX, looking for any `ld.global` before
the `griddepcontrol.wait`.

Examples in the layer:
- `_splitk_finish` waits before reading the workspace; `_c1_finish` reads its correction tables
  (weights) before the wait and the accumulator after it.
- `route_pre_expert` loads the router bias and the norm weights before the wait and the logits
  after it.
- The post-FFN boundary loads the residual stream and its weights before the wait. The post-attention
  boundary wrote the stream several kernels earlier.
- The MoE chain (from bf16): each kernel waits at the start. The first one (route prep) runs without
  PDL.

**Transitivity.** If C waits on B and B waited on A before finishing, then when C's wait returns, A
has already finished. For this to hold, every kernel launched with PDL must execute its `wait`
before it ends. If one did not, it would break the chain for the kernels after it.

**What did not win.** Launching the main projection kernels with PDL (MAIN_PDL) and the attention
kernel A (GDCQ, from the attention work) changed nothing: not on the H100, and not on the H200 for
the attention. The previous GEMV occupies every SM, so the dependent cannot get ahead until it ends.
The real window is short (1-3 µs).

## 7. The split-K workspaces

With few tokens, a projection is almost a GEMV: [T, K] × [K, N] with small T. If each CTA computes a
64-row output tile, the pooler (N = 4,096) gives only 64 tiles for 132 SMs: half the GPU idle, in a
memory-bound kernel. Split-K also cuts the K dimension: each tile is shared among S CTAs, each one
reads 1/S of the weights and adds its part into an FP32 accumulator.

```
           K (8192) ──────────────────────────────▶
 tile n  │ CTA (n,0) │ CTA (n,1) │ CTA (n,2) │ CTA (n,3) │   each: partial = x[:, Kslice] · W[n, Kslice]ᵀ
           └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┘
                 ▼  atomic add (FP32 red, relaxed)
          workspace acc[T, N] (FP32, in L2)
                 │
                 ▼  griddepcontrol.wait (all reds have finished)
          _splitk_finish: out = bf16(acc); acc = 0
```

Three details:
- **The workspace starts and ends at zero.** The finish zeroes it for the next call. That way no
  per-call memset is needed, which would be another kernel. If something skipped the finish, the
  next call would add garbage.
- **One workspace per call site** (qkv, o, pooler, the unpooler slice, unpooler, router). The router
  and the pooler run at the same time on two streams, and with a shared workspace they would mix
  their sums.
- **The finish can be fused.** With ≤8 tokens, the post-FFN boundary reads the unpooler's workspace
  directly, rounds to BF16 as the finish would, and zeroes it (v40). That is one launch fewer.

c1 does the same with compressed weights: the GEMV decompresses in registers (12 bits per weight
instead of 16, 25% fewer bytes) and the finish adds the exact corrections of the few weights that
do not fit the format.

## 8. Where the time goes

From [`time-breakdown.md`](time-breakdown.md) (H200 profile of the final check, v54 `54914d08`).
Average weight of each piece in the score:

| piece | weight | distance from its floor | if it were halved |
|---|---|---|---|
| MoE | 48% | 1.3-2.7 times (2.7 with 256 tokens) | x1.32 |
| projections (c1 + split-K + cuBLAS) | 28% | ~1.4 times with few tokens; ~2 with 128-256 | x1.16 |
| attention | 15.6% | full: 1.5-2.8 times; sliding: 6.5-23 µs, pure latency | x1.085 |
| small kernels (boundaries, routing, others) | 9% | ~15 µs per shape with few tokens | x1.05 |

The floor of each piece is moving its bytes at 4.8 TB/s or doing its compute at 989 TFLOP/s,
whichever takes longer.

What the numbers say:
- The MoE rules. On its own, at 1.5 times its floor, it would give 1.89x.
- With what the other three kernels have within reach ("MoE at 2 times, attention at 2 times, projections at 1.3
  times") the layer gets to ~1.88x. Not even the MoE at 1.5 times its floor plus the small kernels
  at 4 µs per shape (2.23x) reaches rank 40 (2.252x).
- The projections with 128-256 tokens are already close to compute: the pooler with 256 tokens runs
  at ~600 TFLOP/s. The weak one is qkv_gate (~400 TFLOP/s), and that needs a custom GEMM.
- The `f31f6572` attention was 6% faster in its own problem (1.79x in a check), and in the layer it
  shows in decode 1 full (from ~107 to ~100-102 µs). Since it is 15% of the layer, that is ~1%,
  below what a check can tell apart.
- The same with MoE v55: 3% better in its own problem, and in the layer it only shows in verify 16
  sliding and verify 1. The layer's routing is not the one in its problem, and with 256 tokens the
  dynamic queue ties.

## 9. Lessons

**The H100 bench does not judge compute changes.** The bench is the 64 GB H100 of the university cluster,
whose measured DRAM peak is ~1.6 TB/s (not the 3.35 TB/s of the 80 GB part); the judge's H200 has about
4.8 TB/s with similar compute. What hides behind
memory on the H100 (decompressing, decoding, changing the MMA type, changing the CTAs per SM)
becomes the bottleneck on the H200. Between the 27th and the 28th the bench gave the opposite sign
from the judge five times:

| change | H100 bench | H200 judge |
|---|---|---|
| MoE v41: wgmma from 65 tokens | verify 32 −18% | gemm1 and gemm2 +49% with 128 tokens; layer 1.634 against 1.669 |
| c1 with 17-32 tokens and qkv in c1 | decode 32 −10 µs | decode 32 +12/+31 µs; layer 1.653 against 1.676 |
| attention: whole rows per cluster | −30% | +18% |
| attention: PV in FP8 with wgmma | +1.5% | +45% |
| MoE: specialized warps (v51-v54) | tie in verify 32 | gemm1 and gemm2 ~2 times slower |

On the other hand, changes that only remove latency or bytes do carry over: stream-K for qkv with 1
token, the fused finish, the per-weight correction limit, the MoE PDL chain. Rule: the bench is good
for the tests and for latency or byte changes; everything else is measured on the H200 before
stacking anything on top of it.

**Run-to-run noise is ±1.5-3%.** The same version, measured twice, moves that much:
- v41: check 1.6761, submit 1.6579 (−1.1%);
- v45: check 1.6854, submit 1.6754 (−0.6%);
- v36 and v37 with the same attention: 1.6543 and 1.6687;
- v45, v48, v52, v53 and v54 (changes of 1-40 µs on some shapes): 1.6854, 1.6774, 1.6786, 1.6663
  and 1.6684.

A 0.5% improvement does not show in a check. That is why small improvements are judged per shape
on the bench, alternating versions, and stacked before spending a check. And that is why the
submit rule is to beat the upper bound of the mark's 95% CI: otherwise the board can end up with a
worse number than the one it already had.

**Measure before building.** The persistent megakernel (everything in one kernel with grid
barriers) looked like the answer to the ~17 launches. The numbers gave ~+1% net: the barriers cost
almost as much as the gaps they remove, and the MoE weighs too much for the rest to matter. It
was not built.

**The profile lies a little.** With PDL and two streams, a kernel counts as its own the time it
spends waiting in `griddepcontrol.wait`, and the GPU times add up to more than the wall time. To
know what a piece costs you have to remove or change it and look at the wall time, taking care that
the change does not alter the work of the others. The ablations of the finishes changed the router
logits and with them the MoE's work: ±100 µs that meant nothing.
