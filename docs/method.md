# Method: how the work was organized

The work ran from 24 Oct to 30 Oct 2026, in four work blocks. This document covers how it was
organized: the challenge rules, the split by problem, the H100 bench, the quota rules and the tools
that need no GPU. What came out of it is in [`lessons.md`](lessons.md) and
[`timeline.md`](timeline.md).

## The challenge

- **What is measured.** Four ops of the Muse Spark decode layer (plus the prefill op, which I left
  out), each one against SGLang's kernel on an H200.
- **How it is measured.** Each shape is captured as a CUDA graph and replayed with a cold L2,
  alternating with SGLang. The speedup of a shape is the median of the paired ratios. The speedup
  of a problem is the geometric mean over the loads, with a 95% CI of at least ±1.5%.
- **Hard rules.**
  - No host syncs, no memory allocation and no branching on input values inside `kernel_fn`.
  - No caching of anything derived from the data.
  - Weights may only be repacked, never changed in precision.
  - Attention reads every key.
  - The organizers rerun the top entries on neighbouring shapes and read the code, so dispatch goes
    by shape ranges and never by exact shape.
- **Quota.** 20 checks and 10 submits per problem per day. A check runs exactly like a submit but
  does not go on the board. Each run gets 10 minutes, compilation included.
- **The overall board.** It ranks exactly like the `muse_decode_layer` board. I verified this on
  27 and 30 Oct on the top 300 rows. The other three kernels count through what they add to the
  layer.

## The split by problem

```
                     plan, quota, bench, state
                               │
       ┌──────────────┬────────┴──────┬──────────────┐
   bf16_proj      attention          MoE       decode_layer
   (one branch, one worktree and one lab notebook per problem)
```

- **One branch per problem.** Each problem had its own branch, worktree and lab notebook. The
  notebook got an entry at every experiment: what changed, the bench numbers, the judge's numbers
  and the decision. decode_layer composes what the other three win, so it pulled their kernels in
  as they improved.
- **One place for the plan.** A single state document held the plan, the quota left on each
  problem and the bench queue. I handed out spare checks from there before they expired.
- **A written review before each submit that mattered.** That covered the first submit of each
  problem, every new best mark and the freeze. The request was a diff, the evidence and one
  concrete question. The answer was a verdict (submit or don't) with findings graded blocking,
  important or minor. 30 reviews were written. Some checked a claim with a small simulation of
  their own: the mbarrier parities over 5760 executions, the NVFP4 decode bit by bit, the MoE
  item queue under random interleavings.
- **State in files.** Small commits, the notebooks and the state document. Nothing that mattered
  lived only in my head; [`timeline.md`](timeline.md) is a summary of that state document.

## The bench: an H100 that imitates the judge

The judge is a remote H200 with a quota. To iterate without spending it I built a bench on the
H100s of a university cluster ([`scripts/bench/`](../scripts/bench/)).

- **The judge's image.** A Singularity SIF built from the same Docker digest the judge uses (SGLang
  `a55ab7d`, torch 2.14.1, Triton 3.8.0, CUDA 13.0). The cluster has no internet access, so the
  image was built locally and uploaded. That was 16.6 GB at 1-5 MB/s, about 3 hours on the first
  day.
- **The judge's protocol.** `judge.py` repeats the tests of each problem, the static checks, the
  CUDA graph, the cold L2, the alternating pairs and the geometric mean with its 95% CI. An A/A
  test gave between 0.998 and 1.001 on all five problems.
- **Queue over SSH.** `bench run <problem> <file>` packs the file, drops it in a queue on the
  cluster's scratch and, if no worker is running, submits a one-GPU Slurm worker. The worker serves
  the queue serially and releases the GPU after 10 minutes without work.
- **Limits.** 6 H100-hours per day, enforced by the bench itself, and a total limit for the final
  batch. In total I spent 18.74 GPU-hours in 60 jobs
  ([`evidence/h100-hours.jsonl`](../evidence/h100-hours.jsonl), taken from `sacct`).
- **Calibration.** On the first day, one check with the untouched baseline on each problem, and the
  same baseline on the bench. [`evidence/calibration.json`](../evidence/calibration.json) gives
  the H200/H100 factor per shape (68 of 68 paired):

  | problem | mean factor | range |
  |---|---|---|
  | bf16_proj | 0.56 | 0.40-0.96 |
  | attention | 0.79 | 0.71-0.88 |
  | MoE | 0.73 | 0.57-0.97 |
  | decode_layer | 0.70 | 0.59-0.90 |

  Bandwidth-bound shapes sit at ~0.4 and compute-bound shapes at ~1.0. The bench's H100 is the
  64 GB part, with ~1.5-1.6 TB/s measured, so the H200 has about three times its bandwidth.

The bench was useful for correctness and for changes in latency or bytes. For everything else it
often predicted the wrong sign ([`lessons.md`](lessons.md), § 1), and from 27 Oct on the check on
the H200 made the call.

## Quota and submit rules

- **Checks.** Each problem could spend half of its daily quota freely. The rest was handed out
  from the plan before it expired at 01:00 CET.
- **Submit.** It needs a check that passes and beats the problem's own best mark by more than its
  95% CI, plus a review that says submit. In the final batch it needed two checks. The second one
  ran on a copy that changes only a comment, so the judge could not return the cached result.
- **Evidence.** Each run keeps its judge JSON, its profile and its log in
  `evidence/runs/<problem>/<job>` (this repo publishes the JSON and the profile). Each submit that
  reached the board has the git tag `okc/<problem>/<job>` in the private repo. The tag alone does
  not identify the file, because several submits came from candidate copies. The submitted file is
  the one in `kernels/<problem>/` (or `versions/`), and
  [`evidence/provenance.json`](../evidence/provenance.json) gives its source and its sha256.
- **Marks.** A mark counted only after two checks against the judge's JSON: its content_hash
  matches the check's, and the file's sha256 is the one in the repo. A summary in a notebook is not
  proof.
- **No gaming.** When in doubt, don't submit. If it is already on the board, `okc withdraw`.

## Tools without a GPU

The local box has an RTX 2080 (Turing), which cannot measure Hopper. I used it to:
- compile the kernels to sm_90 SASS with NVRTC and Triton's ptxas, to see registers, spills and
  instructions before spending a check (`kernels/*/tools/`);
- emulate on sm_75 the BF16 instructions the 2080 lacks (`-DOKC_EMULATE`) and run the correctness
  tests against the reference;
- brute-force on the CPU the bit layout of the NVFP4 decode
  (`kernels/muse_moe_w4a16/tools/search_*_layout.py`).

## What did not work in the organization

- **Work stopped while waiting for quota.** On 26 Oct, problems that had used their checks sat
  idle until the next day's quota, with work pending that the bench could have taken. That lost
  about 10 hours. Two rules came out of it: waiting for quota is not a reason to stop, and checks
  about to expire are handed out before 01:00.
- **A worker that did not check its GPU.** On 25 Oct, a worker submitted to the interactive queue
  landed on a login node with no usable GPU and swallowed every run it took for hours. Since then
  the worker checks that torch sees the GPU inside the container before it accepts work.
- **Checks outside the plan.** The first block allowed only calibration and bf16. When the bench
  fell behind, 6 attention and MoE checks went out anyway. That was corrected the same night.
