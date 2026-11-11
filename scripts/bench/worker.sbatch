#!/bin/bash -l
# Bench worker: one H100 that drains the queue in $B/queue and exits on its own after
# OKC_IDLE_S seconds without work. `bench` submits it only when there is a queue, with
# --qos set to the first QoS in $OKC_BENCH_QOS that has no worker yet. If $B/paused holds
# today's date (daily H100 hour cap), it finishes the run in progress and exits.
# ACCOUNT, QOS and SCRATCH are placeholders: #SBATCH lines cannot read variables.
#SBATCH --job-name=okc-bench
#SBATCH --account=ACCOUNT
#SBATCH --qos=QOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=SCRATCH/okc/bench/logs/worker-%j.log

set -u
B=${OKC_BENCH_DIR:?}
SIF=${OKC_BENCH_SIF:?}
IDLE_S=${OKC_IDLE_S:-600}
MARGIN_S=1260  # one run (900 s + 300 s grace + copy) has to fit before the 2 h limit

module load singularity/4.1.5
mkdir -p "$B/queue" "$B/running" "$B/done"
start=$(date +%s)
last=$start
deadline=$((start + 7200 - MARGIN_S))
echo "$(date -Is) worker $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
# On a login node nvidia-smi sees the GPU and the container does not: check inside the
# container before taking any run, so that they do not all fail in 3 s.
if ! singularity exec --nv --cleanenv "$SIF" python3 -c \
    "import torch; assert torch.cuda.is_available(); print('torch sees', torch.cuda.get_device_name(0))"; then
  echo "$(date -Is) the container does not see the GPU on $(hostname): exiting without taking runs"
  exit 3
fi

while :; do
  date +%s > "$B/heartbeat.$SLURM_JOB_ID"
  # Before looking at the queue: with an empty queue the worker would wait its 10 minutes.
  if [ "$(cat "$B/paused" 2>/dev/null)" = "$(date +%F)" ]; then
    echo "$(date -Is) daily H100 hour cap reached: releasing the GPU"
    break
  fi
  id=$(ls -1 "$B/queue" | grep -v '\.tmp$' | sort | head -n1)
  now=$(date +%s)
  if [ -z "$id" ]; then
    if [ $((now - last)) -ge "$IDLE_S" ]; then
      echo "$(date -Is) $IDLE_S s with an empty queue: releasing the GPU"
      break
    fi
    sleep 5
    continue
  fi
  if [ "$now" -ge "$deadline" ]; then
    echo "$(date -Is) close to the 2 h limit: $id stays queued for the next worker"
    break
  fi
  mv "$B/queue/$id" "$B/running/$id" 2>/dev/null || continue
  d=$B/running/$id
  lim=$(cat "$d/limit_s" 2>/dev/null || echo 900)
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/okc.XXXXXX")
  echo "$(date -Is) starting $id (limit $lim s)"
  t0=$(date +%s)
  mapfile -t args < "$d/args"
  # With --ncu REGEX on the client: Nsight Compute profile of the first matching kernel.
  ncu_re=$(cat "$d/ncu" 2>/dev/null || true)
  pre=()
  if [ -n "$ncu_re" ]; then
    pre=(/usr/local/cuda/bin/ncu --set full -k "regex:$ncu_re" -c 1 --graph-profiling node -f -o "$d/ncu-profile")
  fi
  # Empty Triton and Inductor caches for every run: compilation counts, as in the judge.
  SINGULARITYENV_TRITON_CACHE_DIR=$tmp/triton \
  SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR=$tmp/inductor \
  SINGULARITYENV_HOME=$tmp \
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 \
  timeout -k 30 $((lim + 300)) singularity exec --nv --cleanenv -B "$B" -B "$tmp" "$SIF" \
    "${pre[@]}" python3 "$d/judge.py" --problem "$d/problem" --submission "$d/submission.py" \
    --output "$d/result.json" "${args[@]}" > "$d/log.txt" 2>&1
  rc=$?
  if [ -n "$ncu_re" ] && [ -f "$d/ncu-profile.ncu-rep" ]; then
    singularity exec --cleanenv -B "$B" "$SIF" /usr/local/cuda/bin/ncu --import "$d/ncu-profile.ncu-rep" \
      --page details > "$d/ncu-profile.txt" 2>&1 || true
  fi
  printf '{"rc": %d, "s": %d, "job": "%s", "node": "%s"}\n' "$rc" $(($(date +%s) - t0)) \
    "$SLURM_JOB_ID" "$(hostname)" > "$d/end.json"
  [ -f "$d/result.json" ] || printf '{"status": "error", "error": "the judge wrote no result (rc %d)"}\n' \
    "$rc" > "$d/result.json"
  rm -rf "$tmp"
  mv "$d" "$B/done/$id"
  last=$(date +%s)
  echo "$(date -Is) finished $id rc=$rc"
done
rm -f "$B/heartbeat.$SLURM_JOB_ID"
