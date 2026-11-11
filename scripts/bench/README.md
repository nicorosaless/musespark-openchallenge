# Bench: an H100 that imitates the judge

All three files live in `scripts/bench/`. `bench` runs on the workstation and sends each run over
SSH to a queue on the cluster. `worker.sbatch` is the Slurm job that serves that queue, and
`judge.py` is what runs inside the judge's image for each run.
[`docs/method.md`](../../docs/method.md) explains why the bench exists and what I learned with it.

```bash
bench run <problem> <file.py> [--no-timing] [--filter 'decode 32'] [--ncu REGEX]
bench fetch <id>    # result of a run that is already queued
bench status        # queue, workers and heartbeat
bench hours         # GPU-hours per job, from sacct
bench stop          # cancel the workers
```

`--no-timing` skips timing, `--filter` keeps only the shapes whose name contains the text, and
`--ncu` runs under Nsight Compute. The other judge options are `--no-tests`, `--reps N`,
`--copies N`, `--no-profile` and `--keep-going` (run every test after a failure). `run` also
takes `--no-wait` (queue and return) and `--max-wait S`.

Set these variables before you use it. In the private repo they held the cluster's values; here
they are published as variables or placeholders:

| variable | what it is |
|---|---|
| `OKC_BENCH_HOST` | SSH alias of the cluster's login node |
| `OKC_BENCH_DIR` | bench directory on the shared scratch (`queue/`, `running/`, `done/`, `logs/`) |
| `OKC_BENCH_SIF` | Singularity image built from the judge's Docker digest; `bench run` waits for `$OKC_BENCH_SIF.ready` to exist |
| `OKC_BENCH_USER` | Slurm user, for `squeue` and `sacct` |
| `OKC_BENCH_QOS` | comma-separated Slurm QoS names for the workers, one worker per QoS (for example a debug QoS) |
| `OKC_NOTIFY` | optional: command that sends a one-line notification |
| `OKC_IDLE_S` | optional, read by the worker: seconds without work before it exits (600) |

In `worker.sbatch`, `ACCOUNT`, `QOS` and `SCRATCH` are placeholders in the `#SBATCH` header.
`bench` overrides the QoS on the `sbatch` command line.

**How it works.**
1. `run` packs the file and the problem and drops them in `$OKC_BENCH_DIR/queue/<id>`.
2. If there are more queued runs than free workers, it submits a `worker.sbatch` (1 GPU, 2 h at
   most) to a QoS from `$OKC_BENCH_QOS` that has no worker yet.
3. The worker checks that the container sees the GPU and serves the queue serially. Each run is
   `singularity exec --nv <SIF> python judge.py`.
4. After 10 minutes without work, the worker exits. If the day's hour cap is reached, the bench
   writes today's date to `$OKC_BENCH_DIR/paused`, the run stays in the queue and `run` exits
   with code 5.
5. `run` waits for `done/<id>/result.json`, copies it to `evidence/bench/<problem>/<id>.json`
   and prints it.

`judge.py` reproduces the protocol in `TASK.md`: the correctness tests, `check_implementation`,
the CUDA graph per shape, the cold L2, the alternating pairs with SGLang, the median of ratios,
the geometric mean over loads and the 95% CI.

`scripts/calibration.py match <okc-status.json>... --bench <bench.json>...` pairs the SGLang
times of an H200 check with those of a bench run and writes `evidence/calibration.json`.
