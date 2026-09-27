#!/usr/bin/env bash
# H1/H2: is the frozen adapter deterministic, and does a zero-residual hybrid choose exactly the
# adapter's action at every step?
set -euo pipefail
cd "$(dirname "$0")/.."
TRACE_ROUNDS="${TRACE_ROUNDS:-25}"
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/H1_H2_parity"
rm -rf "$OUT"; mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for run in a b; do
  Q_AGENT_TRACE_PATH="$OUT/H1_adapter_${run}.jsonl" Q_AGENT_TRACE_AGENT=adapter_0 \
    .MLE/bin/python main.py play --no-gui \
      --agents hybrid_agent/adapter hybrid_agent/adapter hybrid_agent/adapter hybrid_agent/adapter \
      --train 0 --scenario classic --seed 1201 --n-rounds "$TRACE_ROUNDS" --log-dir "$OUT/logs" \
      --save-stats "$OUT/H1_adapter_${run}_stats.json"
done
env -u BOMBERMAN_Q_AGENT_HYBRID_MODEL Q_AGENT_HYBRID_INPUT_DIM=12 \
  Q_AGENT_TRACE_PATH="$OUT/H2_zero_residual.jsonl" Q_AGENT_TRACE_AGENT=hybrid_h3_h8_0 \
  .MLE/bin/python main.py play --no-gui \
    --agents hybrid_agent/hybrid_h3_h8 hybrid_agent/hybrid_h3_h8 hybrid_agent/hybrid_h3_h8 hybrid_agent/hybrid_h3_h8 \
    --train 0 --scenario classic --seed 1201 --n-rounds "$TRACE_ROUNDS" --log-dir "$OUT/logs" \
    --save-stats "$OUT/H2_zero_residual_stats.json"
.MLE/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
def load(name):
    return [json.loads(line) for line in (out / name).read_text().splitlines()]
a, b, h = load("H1_adapter_a.jsonl"), load("H1_adapter_b.jsonl"), load("H2_zero_residual.jsonl")
key = lambda r: (r["round"], r["step"], r["action"])
summary = {
    "rounds": len({r["round"] for r in a}),
    "adapter_decisions": len(a),
    "adapter_run_a_equals_run_b": [key(r) for r in a] == [key(r) for r in b],
    "hybrid_decisions": len(h),
    "hybrid_matches_adapter": sum(key(x) == key(y) for x, y in zip(a, h)),
    "hybrid_equals_own_prior": sum(r["action"] == r["q_agent_action"] for r in h),
}
summary["parity_exact"] = (summary["adapter_run_a_equals_run_b"] and len(a) == len(h)
                           and summary["hybrid_matches_adapter"] == len(a))
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if not summary["parity_exact"]:
    sys.exit("zero-residual parity FAILED")
PY
