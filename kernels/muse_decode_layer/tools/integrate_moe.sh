#!/bin/bash
# Integrate an experts file from the MoE or bf16 kernel into the layer and measure it on the bench.
#
#   kernels/muse_decode_layer/tools/integrate_moe.sh <moe file> <out dir> "<header first line>" [base commit]
#
# Copies the layer's submission.py at <base commit> (default HEAD) to <out dir>/submission.py, runs
# patch_moe.py, and reports anything that still needs a manual fix:
#   - moe_experts against the file's kernel_fn: only the usual adaptations may differ;
#   - moe_prepare against its prepare(): only the weight name may differ;
#   - the experts header comment, whose second line patch_moe drops.
# Then compile_sm90.py, test_cpu.py, a bench run with tests, and an alternated pair without tests
# (base, candidate), with the per-shape deltas. NO_BENCH=1 stops before the bench. Run it from the
# repository root.
set -euo pipefail
src=$1; out=$2; title=$3; base=${4:-HEAD}
W=$PWD
SIF=/data/okc/sglang-a55ab7d.sif
mkdir -p "$out/base"
git show "$base:kernels/muse_decode_layer/submission.py" > "$out/base/submission.py"
cp "$out/base/submission.py" "$out/submission.py"
sha256sum "$src"
python3 kernels/muse_decode_layer/tools/patch_moe.py "$out/submission.py" "$src" "$title"

echo "== moe_experts vs kernel_fn (expected: the ids/wts/x lines and n_local, start)"
diff <(sed -n '/^def kernel_fn(data: Inputs, state)/,$p' "$src" | sed 1,4d) \
     <(sed -n '/^def moe_experts(x, ids, wts, state/,/^# ------/p' "$out/submission.py" | sed 1,7d) || true
echo "== moe_prepare vs prepare (expected: the def line, docstring and post_norm_w -> post_expert_norm)"
diff <(sed -n '/^def prepare(weights: Weights):/,/^# K slices/p' "$src") \
     <(sed -n '/^def moe_prepare(weights):/,/^# K slices/p' "$out/submission.py") || true
echo "== experts header (fix its second line by hand if it lost one)"
n=$(grep -n "^# Experts: " "$out/submission.py" | cut -d: -f1)
sed -n "$n,$((n + 6))p" "$out/submission.py"

echo "== compile_sm90.py"
singularity exec -B "$W:/w" -B "$out:/o" $SIF python3 /w/kernels/muse_decode_layer/tools/compile_sm90.py /o/submission.py 2>&1 \
    | grep -v "^ok" || echo "all ok"
echo "== test_cpu.py"
singularity exec -B "$W:/w" -B "$out:/o" $SIF python3 /w/kernels/muse_decode_layer/tools/test_cpu.py /o/submission.py \
    > "$out/cpu.txt" 2>&1 || true
echo "ok: $(grep -c '^ok' "$out/cpu.txt")"; grep -v '^ok' "$out/cpu.txt" | tail -5 || true

[ "${NO_BENCH:-0}" = 1 ] && { sha256sum "$out/submission.py"; exit 0; }
echo "== bench with tests"
scripts/bench/bench run muse_decode_layer "$out/submission.py" --no-profile 2>&1 | grep -E "status|tests|speedup x|FAIL|Traceback" || true
f=$(ls -t evidence/bench/muse_decode_layer/*.log | head -1 || true)
scripts/bench/bench run muse_decode_layer "$out/base/submission.py" --no-tests --no-profile 2>&1 | grep -E "speedup x" || true
g=$(ls -t evidence/bench/muse_decode_layer/*.log | head -1 || true)
python3 - "$f" "$g" <<'EOF'
import re, sys
def load(f):
    rows = {}
    for l in open(f):
        m = re.match(r"\[shape\] (\w+ \d+) \| .*layer (\w+).*base ([\d.]+) µs, sub ([\d.]+) µs", l)
        if m:
            rows[(m[1], m[2])] = float(m[4])
    return rows
a, b = load(sys.argv[1]), load(sys.argv[2])
for k in a:
    print(f"{k[0]:10s} {k[1]:8s} new {a[k]:7.1f} base {b.get(k, float('nan')):7.1f} {a[k] - b.get(k, 0):+6.1f}")
EOF
sha256sum "$out/submission.py"
