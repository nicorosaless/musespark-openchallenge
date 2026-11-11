# muse_decode_layer: how to read the kernel

`kernel.py` is the file to read: the same kernels as `submission.py` with dead experiment paths
removed and the docstrings fixed. `submission.py` is the exact file the judge scored (sha256 in
[`evidence/provenance.json`](../../evidence/provenance.json)). Line numbers for every symbol are in
[`MAP.md`](MAP.md).

`kernel.py` is 3,347 lines against 4,509. It drops the wgmma verify kernel `verify_wg` (544 lines,
never selected), the experiment switches in Python and CUDA that are fixed in the scored
configuration, the local-test and CPU-interpreter paths and the unused helpers carried over from
the other three kernels. It was checked offline after the judge closed: the same host-side decisions of
`prepare()` and `kernel_fn()` in 350 calls (7,080 records over five weight and fallback variants),
identical sm_90a PTX and CUBIN for all 7 (CUDA source, `-D` set) pairs, identical sm_90 PTX and
CUBIN for all 101 launch configurations of the 8 Triton kernels whose text changed, and 30 of 30
checks on a local RTX 2080 (routing, c1 projections, post-FFN boundary). The attention and MoE GEMM
paths need sm_90 and rest on the PTX comparison.

`submission.py` is an exact copy of submit `b76b7763d6eb45b6`: **1.675x** on the H200, rank 280 when
it went in; sha256 `93600d5e…`, the same content_hash as check `c511c9ea` (1.685x). Commit
`af6dde2` (v45) on `feature/P-2610-okc-decode-layer`.

**Start with [`concepts.md`](concepts.md).** It is the conceptual document for this kernel, written
for the final version: what the layer computes step by step, what pieces it is made of, the kernel
flow, the two streams, the PDL chain, the split-K workspaces, the time breakdown and the lessons of
the block. Then come back here for the code.

**Why this layer is the one that matters.** The overall ranking of the challenge follows the same
order as this board (checked on 27 and 30 Oct; [`docs/method.md`](../../docs/method.md)). The other
three kernels count for what they contribute here.

**What changed from v41 to v45.**
- **v44, per-weight limit on c1 corrections.** Each weight uses the smaller of `C1_CAPS = (8, 16)`
  that fits its corrections; the finish reads fewer columns where 8 is enough.
- **v45, the MoE with ≤ 32 tokens chained with PDL** (work from the bf16_proj problem): route prep → gemm1
  → swiglu → gemm2 → combine, each one launched before the previous one ends and waiting with
  `griddepcontrol.wait` before reading anything. It wins in decode 1, decode 16 full and verify 1.

**What changed from v35 to v41.**
- **v36, the attention candidate with `PSMEM` and `cl`** (page queue in shared memory in
  `verify_ws`, and `full_fa2` in clusters for the decode sliding layers with 1-2 requests).
- **v40, the unpooler finish inside the boundary.** With ≤ 8 tokens, `_ffn_boundary_kernel` reads
  the unpooler's FP32 workspace, rounds it to BF16, zeroes it and goes on with the norm: one launch
  fewer.
- **v41, bf16's v11b.** qkv_gate with 1 token through stream-K (`_c1_gemv_sk`), cuBLAS NN
  (`torch.mm(x, wt)`) for pooler and unpooler with many tokens, and `C1_CAP = 16`.

**What changed from v26 to v35 (+4%).**
- **v35, MoE v24** ([`../muse_moe_w4a16/`](../muse_moe_w4a16/)) replaces v12. It is the largest
  contribution: it lifts every shape, verify 16/32 most of all.
- **v28 and v33, bf16's c1 in the layer.** The pooler with 1-16 tokens, o and the unpooler read the
  losslessly compressed weights of bf16's v10 (`_c1_*`, `compress_c1`). `prepare()` checks the
  round trip bit for bit; v33 uses the tensor-core variant (`_c1_gemm`) in the full unpooler with
  9-16 tokens.
- **v29, PDL in the routing and in the boundary.** They load the weights before
  `griddepcontrol.wait`, overlapped with the tail of the previous kernel.
- **v31, routing in two programs per token.** One routes and the other normalizes, in parallel.

**What changed in v26.** The routing (sigmoid, top-8 with bias, normalization) is no longer
SGLang's: my own kernel, `_route_pre_expert_kernel`, does it, and in one program per token it
combines the routing, the local ids and the pre-expert norm. The side stream only carries the
router.

**What it does.** A whole Muse Spark decoder layer: qkv_gate projection, attention, o projection,
norms with residual, router, top-8, pooler, MoE experts, unpooler and final norm. `submission.py`
is about 4400 lines, but almost all of it is **composition**: the kernels of the other three problems,
copied as they are, plus SGLang ops where mine do not win yet.

**Where each section comes from.** The module docstring lists them in the order the layer runs. It
is the best entry point.
- Attention: the attention candidate with `PSMEM` and `cl`, later than `3c35c84b` and earlier than
  `f31f6572` (see [`../muse_attention_decode/versions/`](../muse_attention_decode/versions/)). It
  carries the CUDA kernels A for decode full (`full_fa2`) and verify full (`verify_ws`), and `cl`
  with clusters on the decode sliding layers with 1-2 requests. No path to FA3 is left.
- Projections with ≤ 32 tokens and router: bf16's v9 split-K ([`../muse_bf16_proj/`](../muse_bf16_proj/)),
  with the reduction in a second kernel with PDL and a defensive `gdc_wait()`. Since v28, pooler, o
  and unpooler with few tokens read the weights in bf16's v10 c1 format.
- Experts: MoE v24 ([`../muse_moe_w4a16/`](../muse_moe_w4a16/), all CUDA) for any number of
  tokens. SGLang's Marlin path disappeared in v20.
- Router: the FP32 weights as BF16 hi + lo in bf16_proj's split-K, for any number of tokens.
  SGLang does the same above 8 tokens; below that it uses an FP32 `F.linear` that runs at ~5% of
  the bandwidth.
- New in this kernel: `_route_pre_expert_kernel` (routing, local ids and pre-expert norm, in two
  programs per token) and `_ffn_boundary_kernel` (post-FFN norm reading the unpooler's slice
  directly, without the zero-padded copy).
- The router runs on a side stream, overlapped with the pooler. Each split-K call site has its own
  workspace so they do not overwrite each other.

**Reading order.** Read `kernel.py`; line numbers in [`MAP.md`](MAP.md). The CUDA (attention and
MoE) is in `cuda/`.
1. The module docstring.
2. `kernel_fn` (at the end): the layer step by step, with the dispatch thresholds (`SMALL_M_MAX`,
   `SLICED_UNPOOL_MAX_TOKENS`).
3. The three new kernels above.
4. `prepare`: what is repacked once.

**After the mark (final batch, 30 Oct).** v53 (PDL up to 16 tokens) gave 1.666x and v54 (v53 with
MoE v55) 1.668x: both within the noise of the mark, no submit. v55's dynamic queue wins in the MoE
problem and not in the layer, because the layer's routing spreads the experts more and the static
split is already balanced.

**What is missing for the top 40 (2.31x on 1 Nov, +38%).** The persistent fusion o-proj → boundary
→ pooler/router → pre-expert was studied and discarded on 27 Oct (~+1% with a lot of risk; see [`concepts.md`](concepts.md),
§ 9). What remains large
is in the pieces: the MoE with 128-256 tokens (75% of verify 16/32) and the full attention in
verify (313 µs with 256 tokens against a ~136 floor). The breakdown by shape and component is in
[`time-breakdown.md`](time-breakdown.md), generated by [`tools/breakdown.py`](tools/breakdown.py) from
the judge's profile.

**Tools** ([`tools/`](tools/)): `patch_moe.py` and `patch_attn.py` transplant the MoE and attention
kernels into the layer; `integrate_moe.sh` runs `patch_moe.py` on an experts file, then the CPU
tests and a bench run; `compile_sm90.py` compiles the CUDA kernels for sm_90 without a GPU;
`test_cpu.py` tests the dispatch logic on the CPU.
