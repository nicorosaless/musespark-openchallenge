# Lessons, with their numbers

Each lesson points to its evidence: a judge check (`evidence/runs/<problem>/<job>.json`), a bench
run (`evidence/bench/<problem>/`) or a kernel's README.

## 1. The H100 bench gave the wrong sign on the H200

The bench imitated the judge well on correctness and on order of magnitude. But for changes that
trade bytes for instructions, touch the MMA path or split the work differently across CTAs, the
H100 and the H200 gave opposite signs or very different magnitudes. Of the 20 changes that can be
paired, 11 flipped sign ([`evidence/bench_vs_h200.csv`](../evidence/bench_vs_h200.csv)):

| change | bench (H100) | judge (H200) |
|---|---|---|
| MoE v41: wgmma in both GEMMs | verify 32 −18% | total 1.733x against 1.818x for the base |
| MoE v51-v54: specialized warps (decode producer, wgmma consumer) | tie on verify 32 | gemm1 324-446 µs against 155-197: 2x slower |
| layer v42-v43: bf16 c1 (compressed weights) with 16-32 tokens | decode 32 −15 / −12 µs | decode 32 +12 / +31 µs |
| attention `CLW`: whole 128 B rows shared across the cluster | −30% | +18% |
| attention `PVA`: 3 CTAs per SM | −1% | +12% |
| attention: PV in FP8 wgmma (same sign, different scale) | +1.5% | +45% |

[`charts/bench_vs_h200.png`](../charts/bench_vs_h200.png) plots each case with its source.

**Why.** The H200 has about three times the bandwidth of the bench's 64 GB H100 (4.8 against
~1.6 TB/s measured) with similar compute. What hid behind
memory on the H100 becomes the bottleneck on the H200. On the H100, DRAM row efficiency dominates;
on the H200, how many CTAs fit per SM and how long each block's dependency chain is.

**What did carry over:** changes that only remove latency or bytes, such as the stream-K of
qkv_gate, the limit on corrections per weight in c1 or the split-K closing kernel chained with PDL.

**Rule that came out of this:** the bench is for correctness and for changes in latency or bytes.
Any other change needs a check on the H200 before anything is built on top of it.

## 2. Noise rules: ±1.5-3%

The judge's 95% CI never goes below ±1.5%, and runs of the same code differ by ±1.5-3%.
- **The same code scores differently.** MoE v24 scored 1.812x in its check and 1.811x in the
  submit. Attention s4 scored 1.798x and 1.795x in two checks and 1.801x in the submit.
- **A submit that does not improve.** Attention submit `d0139e27` landed at 1.687x, below the
  previous mark (1.691x).
- **Gains that were not submitted:**
  - bf16 v11, +1.1% (1.275x [1.256; 1.294]) against v10 at 1.261x [1.242; 1.279];
  - decode_layer v53 (1.666x) and v54 (1.668x) against the mark of 1.675x.

**Rule:** submit only if a check passes and beats the top of the mark's 95% CI. In the final batch
two checks were required: the second on a copy that changes only a comment, because the judge
returns the cached result when the content is identical. Resubmitting the same code to "roll the
dice" is not allowed.

## 3. With few tokens, everything is bandwidth or instruction issue

With 1-32 tokens there is no compute to do: the work is reading weights.
- **Read fewer bytes.** bf16 v10 compresses the BF16 weights losslessly (12 KB instead of 16 per
  64×128 block) and decompresses them in registers. It went from 1.231x to 1.261x, and the pooler
  at m=1 runs below the time it takes to read the uncompressed BF16 weights
  ([`kernels/muse_bf16_proj/`](../kernels/muse_bf16_proj/)).
- **Issue fewer instructions per weight.** On the H200, the MoE's NVFP4 GEMMs are bound by
  instructions, not by memory:
  - MoE v8 reorders the bits of each word in `prepare()` so the E2M1 to BF16 decode costs 11
    operations instead of 15: **+17%** from a single change;
  - v23 takes it to 10 with a multiplication
    ([`kernels/muse_moe_w4a16/README.md`](../kernels/muse_moe_w4a16/README.md));
  - v20 moves the indices to `int` and removes every spill.

  On the bench's H100 these changes barely showed.
- **Counting instructions does not predict time.** In the Triton MoE, going from 6.3 to 4.7
  instructions per value changed nothing: with an IPC of ~1.2-1.8 out of 4, the kernel was
  stalled on waits. You need to know what limits a kernel before you optimize it.

## 4. PDL: small, steady and almost free

Programmatic dependent launch lets a kernel start before the previous one finishes: it loads its
weights and waits on `griddepcontrol.wait` before it reads what the previous kernel writes.
- **bf16 v9.** The split-K closing step went from "counter plus the last CTA writes" to a second
  kernel launched with PDL: between 0.2 and 0.8 µs less per shape. Its review went through the
  PTX memory model to check that both patterns were correct.
- **MoE with up to 16 tokens.** route → gemm1 → swiglu → gemm2 → combine, chained. The bf16 work
  built it, and it went into the layer (v45) and into MoE v55.
- **The trap.** With PDL, write visibility is transitive only if every link waits before it reads.
  The [MoE README](../kernels/muse_moe_w4a16/README.md) goes through the case of the dynamic
  queue's counters.

## 5. Warp specialization is not free: it depends on what limits

- **It won in attention.** `verify_ws` has a producer warpgroup (`cp.async`) and a consumer one
  (wgmma and softmax), coordinated with mbarriers and with registers split by `setmaxnreg`. With
  it, verify full stopped being the slowest shape and attention went from 1.652x to 1.691x.
- **It lost in MoE.** The same pattern (v51-v54) was 2x slower. Only 8 warps per SM decoded NVFP4,
  against 16 in v24, and that decode was the bottleneck. Overlapping with the MMA does not help
  when the MMA never limited.

wgmma against `mma.sync` follows the same logic. When each fragment comes out of a decode in
registers, `mma.sync` with the operand already in registers (and no shared memory) beat wgmma in
every variant ([`kernels/muse_moe_w4a16/what-didnt-win.md`](../kernels/muse_moe_w4a16/what-didnt-win.md)).

## 6. What wins on one op may not win in the layer

The dynamic item queue of MoE v55 gave +2.6% on the MoE board and improved all six shapes. Inside
the layer (v54) it moved nothing (1.668x against 1.675x). The layer's routing spreads the experts
more, so the static split was already balanced. Since the overall board follows the layer board,
what counts is what wins inside the layer.

## 7. Measure before building

- **The fixed cost.** The bf16 work measured a kernel that only reads N bytes with a cold L2: the
  line intercepts at ~0.2-0.8 µs, and my kernels sat 1.6-3.0 µs above it. That said how much
  could be gained by removing fixed cost and how much required reading fewer bytes.
- **Fusions dropped on an estimate.** In decode_layer, the persistent fusion
  o-proj → boundary → pooler/router → pre-expert was estimated at ~+1% with high risk and was not
  built.
- **Compile without a GPU.** Registers, spills and sm_90 SASS were inspected locally before
  spending a check.

## 8. On organization

- **Waiting for quota is not a reason to stop.** It took an explicit rule, and a check every 2
  hours that no problem sat idle with work pending.
- **Written reviews found real issues.** For example:
  - a docstring that said FP16 where the code already used E4M3;
  - a `const __restrict__` on a buffer that is written;
  - partial test coverage in attention's window branch.

  They also settled questions that tests would not have caught, such as the mbarrier parities or
  the NVFP4 decode bit by bit.
- **Evidence in files.** Every mark was checked against the judge's JSON and the file's sha256,
  not against a summary of it.
