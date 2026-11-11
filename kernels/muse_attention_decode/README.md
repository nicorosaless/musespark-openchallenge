# muse_attention_decode: how to read the kernel

`kernel.py` is the file to read: the same kernels as `submission.py` with dead experiment paths
removed and the docstrings fixed. `submission.py` is the exact file the judge scored (sha256 in
[`evidence/provenance.json`](../../evidence/provenance.json)). Line numbers for every symbol are in
[`MAP.md`](MAP.md).

`kernel.py` is 1,685 lines against 4,248. It drops `KERNEL_A_WG_CUDA_SOURCE` (the wgmma verify
kernel, which no `TUNING` entry selects any more), the `#if` branches of about 25 experiment
switches in `verify_ws` and `full_fa2` that are off in the scored configuration, the reporting
helpers and the history comments. It was checked offline after the judge closed: the same
host-side decisions in 1,472 (shape, pass) calls, identical sm_90a PTX for all 16 (CUDA source,
`-D` set) pairs those calls compile, identical sm_90 PTX for all 54 compile keys of `_attn_kernel`
and identical syntax trees for the other Triton kernels, and the 16 `TASK.md` tests, cut down to
run in Triton's CPU interpreter, bit-identical to `submission.py`.

`submission.py` is an exact copy of submit `28dd5b809b434132` (30 Oct, final batch): **1.801x** on
the H200, 95% CI [1.774, 1.828]. Its sha256 is `1c2cfcc3…` and its content_hash `e4a15668…`, the
same as check `d031d282` (1.798x). It is `muse_attention_decode/candidatos/cudec_r12c_s4_9da0607.py`
at `40e89b4` on `feature/P-2610-okc-atencion`.

**s4 compared with the previous mark** (`f31f6572`, 1.760x, `cudec_r12_comb128w4_2bd0c83.py`). It
changes one configuration line: `TUNING[("sliding", "verify")]["cuda"] = "ws"`. With it, the
sliding verify layers move from the Triton kernel to `verify_ws` with `WINDOW = 2048`. Verify
1/16/32 sliding drop by 10-15%: from 10.0/17.4/23.9 µs to 9.0/15.3/20.5 in the submit (9.0/14.8/20.9
in check `d031d282`). The scored file also carries two minor cleanups: `mb_wait` without
`CLW`, and `_lastm_counters` only under `LASTM`. Both concern experiment switches that are
off in the scored configuration. The window branch of `verify_ws` (`t < lo`) and the `ROPE = 1`
path have only run on the judge, because the local tests do not go past the window.

Before s4, `f31f6572` had added this on top of `3c35c84b` (1.691x, `cudec_vws1s4nw_3986153.py`):
- **`cl` with `KVP = 1`** for sliding with batch ≤ 2: a single kernel with clusters and DSMEM, two
  clusters per request (one per KV head). See [`concepts.md`](concepts.md), § 5.
- **`KVEARLY` and `EPIW`**: the prologue and epilogue of `full_fa2` moved earlier.
- **Combine with `COMBINE_SP = 128` and `COMBINE_WARPS = 4`**: 128 splits per pass.

The rest of this page describes the base design, which this version leaves unchanged.

**Start with [`concepts.md`](concepts.md).** It is the conceptual document for this kernel, written
for the final version: split-KV and the combine, `full_fa2`, `verify_ws` with producer and consumer,
`cl` with clusters, what did not win on the H200, and the gap to the floor. Then come back here for
the code.

**What is new: a third kernel A, `verify_ws`, for verify on full layers**
(`KERNEL_A_WS_CUDA_SOURCE`, readable in [`cuda/verify_ws.cu`](cuda/verify_ws.cu)). It is FA3 at
small scale, with specialized warps:
- **Producer and consumer per CTA.** Each 256-thread CTA has a producer warpgroup (only one warp
  copies, with 16 B `cp.async`) and a consumer warpgroup (QKᵀ with E4M3 wgmma, softmax, and PV with
  `mma.sync`).
- **Synchronization with mbarriers.** They coordinate through `full[s]` and `empty[s]` per stage,
  four stages, with no `__syncthreads` in the loop.
- **Registers split with `setmaxnreg`.** The producer drops to 24 and the consumer rises to 104, so
  4 CTAs still fit per SM.

A simulation over 5760 executions checked the mbarrier parities. It is the most
instructive code in the repo on Hopper producer/consumer synchronization. The `TUNING` entries
choose the CUDA kernel for each layer type and load.

**Precision of the CUDA kernels.** Since `13e72a6`, the QKᵀ product of the CUDA kernel is in E4M3
(`mma_k32_e4m3`, in [`cuda/full_fa2.cu`](cuda/full_fa2.cu)), with q rounded to E4M3 at unit RMS,
the same as the reference and FA3. P and V stay in FP16.

**What it does.** One attention step of decode (1 token) or verify (8 draft tokens) over a paged
FP8 cache with two KV heads. The whole layer runs in one kernel: QK-norm, RoPE, writing the new
keys and values into the FP8 cache, the attention, and the output norm-gate. SGLang splits this
into several kernels: norms, RoPE, cache write, FA3 and gate.

**Three kernels A, depending on the layer.**
- **Sliding layers** (window of 2048, 46 of the 62 layers): the Triton kernel `_attn_kernel`.
- **Verify on full layers:** `verify_ws` (above).
- **Decode on full layers** (whole context, up to 1M keys): a CUDA kernel (`KERNEL_A_CUDA_SOURCE`,
  readable in [`cuda/full_fa2.cu`](cuda/full_fa2.cu)), compiled in-process with NVRTC and launched
  with `cuLaunchKernelEx`. It exists because the Triton kernel could not overlap the page-table load
  with the K/V load: each block paid for both in series. The CUDA kernel has an explicit `cp.async`
  pipeline, with the pages `2·STAGES-1` blocks ahead and K/V `STAGES-1` blocks ahead. Decode full
  goes from 24.2 / 116.7 / 226.5 µs (`83ae906c`) to 19.9 / 108.2 / 200.2 in submit `7efd1a8b`
  (B 1/16/32).

**Why it beats SGLang.**
1. **Fusion.** It saves launches and round trips to HBM. This matters most on the sliding layers,
   where fixed cost dominates.
2. **Split-KV that fills the GPU.** With 2 KV heads there are very few programs for 132 SMs. The
   keys are divided into splits sized to fill exactly one wave of resident CTAs, using the measured
   occupancy of the compiled kernel. A separate combine merges the splits by their log-sum-exp and
   launches early with PDL.
3. **Packed heads in decode.** The two KV heads of a pair share one 16-row tile.
4. **Masks only at the edges**, in verify.
5. **Explicit pipeline in decode full** (the CUDA kernel).

**Reading order.** Read `kernel.py`; line numbers in [`MAP.md`](MAP.md). The CUDA is in `cuda/`.
1. The module docstring and the `TUNING` table. Note which entries set `cuda`.
2. `kernel_fn` (at the end): the dispatch between Triton and CUDA and the split computation.
3. `_attn_kernel` (Triton): the loop over key blocks.
4. `KERNEL_A_CUDA_SOURCE`: the same algorithm in CUDA. Focus on the pipeline prologue, on
   `issue_pages` / `issue_kv` and on the `wait_group` of each iteration. [`concepts.md`](concepts.md)
   walks through the same pipeline.
5. `KERNEL_A_WS_CUDA_SOURCE` (`verify_ws`, readable in `cuda/verify_ws.cu`): the producer warp, the
   mbarrier ring and the consumer warpgroup's wgmma loop.
6. `_combine_kernel`: the merge of the split partials and the norm-gate.

**What is missing for the top 40 (2.76x on 1 Nov).** Verify 16/32 full is still at 169/321 µs, far
from the ×0.7 target (118/231). [`gap-to-top40.md`](gap-to-top40.md) puts each shape against its
bytes and compute floor. The next ideas were to copy V in bulk and to convert V in the
producer.

**Versions in this directory.** Each one was a submit and adds one piece of the final design.

| file | submit | speedup (95% CI) | what it introduces |
|---|---|---|---|
| [`versions/1_83ae906c_triton.py`](versions/1_83ae906c_triton.py) | `83ae906cdb87463c` | 1.588x (1.565-1.619) | all Triton: split-KV, block-diagonal tiles for decode, combine with PDL |
| [`versions/2_7efd1a8b_full_fa2.py`](versions/2_7efd1a8b_full_fa2.py) | `7efd1a8b2aa74b53` | 1.652x (1.627-1.676) | `full_fa2`: CUDA kernel A for decode full (`cp.async` with pages ahead, QKᵀ in E4M3) |
| [`versions/3_3c35c84b_verify_ws.py`](versions/3_3c35c84b_verify_ws.py) | `3c35c84b52c14b79` | 1.691x (1.666-1.717) | `verify_ws`: verify full with specialized warps, mbarriers and `setmaxnreg` |
| [`versions/4_f31f6572_clusters.py`](versions/4_f31f6572_clusters.py) | `f31f65728961447b` | 1.760x (1.733-1.792) | `cl` (sliding with batch 1 in clusters with DSMEM), `PSMEM`, `KVEARLY`, `EPIW`, single-pass combine |
| `submission.py` | `28dd5b809b434132` | **1.801x** (1.774-1.828) | verify sliding through `verify_ws` (one `TUNING` line) |
