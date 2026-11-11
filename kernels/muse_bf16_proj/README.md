# muse_bf16_proj: how to read the kernel

`submission.py` is an exact copy of submit `b34454802aa64dc4`: **1.261x** on the H200, 95% CI
[1.242, 1.279], rank 389 of 1104 on entry (27 Oct, 02:36); sha256 `099a099d…`, content_hash
`8c8c75d9…`, the same as check `03bfc6c7`. It is `muse_bf16_proj/submission_v10.py` at `b0e4553`
on `feature/P-2610-okc-bf16-proj` (tag `okc/muse_bf16_proj/b34454802aa64dc4`). The previous mark, `93e8707c` (v9), gave 1.231x.

`kernel.py` is the file to read: the same kernels as `submission.py` with dead experiment paths
removed and the docstrings fixed. `submission.py` is the exact file the judge scored (sha256 in
[`evidence/provenance.json`](../../evidence/provenance.json)). Line numbers for every symbol below
are in [`MAP.md`](MAP.md).

What `kernel.py` drops: the `SPLITK_PDL` switch (always on in the scored file) and three switches
that existed only for local tests on the CPU interpreter or older GPUs (`C1_GDC`, `C1_ASM`,
`C1_DOT_FP32`), with their branches. It was checked offline after the judge closed: the same host-side decisions in 898
`kernel_fn` cases, identical sm_90 PTX for all 14 Triton launch specializations, and 21 of 21
cases on a local RTX 2080 against an FP64 reference.

**Start with [`concepts.md`](concepts.md).** It is the conceptual document for this kernel (revised
on 30 Oct with the final numbers). It explains why a BF16 GEMV is memory-bound, the c1 format
with a real example from the pooler, split-K with a PDL finish, everything that didn't win on the
H200 with its numbers, and the MoE with few tokens. Then come back here for the code.

**What changed in v10: losslessly compressed BF16 weights ("c1" format).** Reading fewer bytes is
the only way to go below the bandwidth floor (see [`roofline.md`](roofline.md)).
- **The format.** Each block of 64×128 weights takes 12 KB instead of 16. There are two
  sign|mantissa planes, one plane of 4-bit exponent codes relative to the block's maximum E, and a
  per-row table with the few weights that do not fit (zeros, or exponents far below E).
- **The arithmetic.** The output adds Σ(v − d0)·x for those corrections.
- **The check.** `prepare()` compresses, decompresses and compares bit by bit. If a layer does not
  fit (more than `C1_CAP = 8` corrections per row), it stays on v9.
- **Where it is used.** With m=1 on pooler, o and unpooler (`_c1_gemv`, decoding in registers with
  PTX `prmt`/`lop3`), and with 2-16 tokens on pooler (`_c1_gemm`, tensor cores).
- **The decode check.** A simulation of the PTX decode on 262,144 exhaustive cases found no
  failure.

**What v9 already had.** The split-K reduction is no longer done with "counter + the last CTA
writes and resets" inside one kernel. It is now two kernels:
1. `_proj_kernel_red` runs the usual loop and adds its FP32 partial to the workspace with relaxed
   atomics.
2. `_splitk_finish`, launched with PDL (programmatic dependent launch), waits in
   `griddepcontrol.wait` for the first one to finish, writes the output and leaves the workspace at
   zero.

It saves between 0.2 and 0.8 µs per shape. The module docstring of `submission.py` still says
"one Triton kernel per call", which is stale; `kernel.py` has the corrected one.

**What it does.** `out = x @ W[:n].T` for the model's dense projections, with few tokens (m from 1
to 256). With small m there is no compute to speak of: the problem is reading the weights from HBM
as fast as possible.

**Why it beats SGLang.** With m ≤ 32, cuBLAS launches a split-K and then a second kernel that sums
the parts (`splitKreduce`), and the FP32 router goes through 3 to 5 kernels. Here the main work is
a single launch:
1. The K axis is split across CTAs so that all SMs read weights at the same time.
2. Each CTA adds its partial to an FP32 workspace with atomics. A second kernel, chained with PDL,
   writes the output and leaves the workspace at zero.
3. `prepare()` repacks the weights into contiguous tiles: each loop step reads one contiguous block
   of memory instead of scattered rows.
4. The MMA is "swapped": the rows of W take the M side of the tensor core and the few tokens the N
   side, so no hardware is wasted on a tiny m.
5. The FP32 router is split once into BF16 high + low, and two BF16 MMAs run on the same FP32
   accumulator. It is what SGLang does, in one kernel.

Where there are already enough output tiles to fill the GPU (m > 32, or the `o` layer), K is not
split and SGLang's GEMM is called as is.

**Reading order.** Everything is in `kernel.py`; line numbers in [`MAP.md`](MAP.md).
1. The module docstring: the whole design, one paragraph per idea.
2. `kernel_fn` (at the end): the dispatch by m range and by shape.
3. `prepare` and `pack`: what is done once, outside the timed region.
4. `_split` and `_needs_split`: how many CTAs per tile, and why.
5. `_proj_kernel_red`: the load loop + `tl.dot`, with the relaxed `atomic_add` at the end.
   `_proj_kernel` is the same loop without split.
6. `_splitk_finish` and how `_project` launches it with `launch_pdl=True`.

**What is missing for the top 40 (1.59x on 1 Nov).** See [`roofline.md`](roofline.md): verify
16/32 (m 128/256) still run on cuBLAS at ~1.1x, and four custom GEMMs do not beat it on the H200
([`what-didnt-win.md`](what-didnt-win.md)).

**Versions in this directory.**

| file | submit | speedup | what it introduces |
|---|---|---|---|
| [`versions/v03_4eb830e7.py`](versions/v03_4eb830e7.py) | `4eb830e71aa0475c` | 1.207x | split-K in Triton with swapped MMA, hi/lo BF16 router |
| [`versions/v09_93e8707c.py`](versions/v09_93e8707c.py) | `93e8707c5082480b` | 1.231x | split-K finish in a second kernel with PDL |
| `submission.py` / `kernel.py` (v10) | `b34454802aa64dc4` | **1.261x** | losslessly compressed weights (c1) |
| [`versions/v11b_unsubmitted.py`](versions/v11b_unsubmitted.py) | check `54006737` | 1.271x | stream-K on qkv_gate with m=1; +0.8%, inside the 95% CI, not submitted. It has been in the layer since the layer's v41 |

[`experiments/`](experiments/) keeps two readable negative results:
[`gemm_ws.py`](experiments/gemm_ws.py) (a warp-specialized GEMM in Gluon: TMA, mbarrier,
asynchronous wgmma; it ties cuBLAS on qkv_gate m=128 and loses everywhere else) and
[`gluon_c1.py`](experiments/gluon_c1.py) (c1 with wgmma RS). [`tools/`](tools/) has the c1
prototype (`c1.py`), its exact CPU tests (`test_c1.py`, `test_cpu.py`) and `sass_c1.py`, which
compiles Triton and Gluon kernels to sm_90 SASS without a GPU.
