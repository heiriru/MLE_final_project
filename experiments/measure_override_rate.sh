#!/usr/bin/env bash
# How often does each trained stage play something other than the q-agent? 50 traced rounds per
# stage.

set -euo pipefail
cd "$(dirname "$0")/.."
ROUNDS="${ROUNDS:-50}"; SEED="${SEED:-4801}"
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/override_rate"
rm -rf "$OUT"; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1 Q_AGENT_H12_INPUT_DIM=14 Q_AGENT_H11_INPUT_DIM=14 Q_AGENT_H10_INPUT_DIM=6
export BOMBERMAN_Q_AGENT_HYBRID_MODEL="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H7_six_inputs_seed1501.pt"
export BOMBERMAN_Q_AGENT_H11_MODEL="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H11_features_seed1501.pt"
export BOMBERMAN_Q_AGENT_H10_MODEL="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H10_dueling_seed1501.pt"
export BOMBERMAN_Q_AGENT_H12_MODEL="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H12_authority_seed1501.pt"

# Plays 50 traced rounds with one stage and checks that a trace was written.
measure () {
  local agent="$1"; shift
  local name="${agent##*/}"
  mkdir -p "$OUT/logs_$name"
  env "$@" BOMBERMAN_SEED="$SEED" Q_AGENT_TRACE_PATH="$OUT/$name.jsonl" Q_AGENT_TRACE_AGENT="$name" \
    .MLE/bin/python main.py play --no-gui \
      --agents "$agent" rule_based_agent rule_based_agent q_agent/q_agent_v1 \
      --train 0 --scenario classic --log-dir "$OUT/logs_$name" --seed "$SEED" \
      --n-rounds "$ROUNDS" --save-stats "$OUT/$name.json"
  [[ -s "$OUT/$name.jsonl" ]] || { echo "no trace written for $agent" >&2; return 1; }
}
measure hybrid_agent/hybrid_h3_h8 Q_AGENT_HYBRID_INPUT_DIM=6 Q_AGENT_RESIDUAL_LIMIT=0.25 Q_AGENT_OVERRIDE_MARGIN=0.25
measure hybrid_agent/hybrid_h10
measure hybrid_agent/hybrid_h11
measure hybrid_agent/hybrid_h12
.MLE/bin/python - "$OUT" > "$OUT/summary.txt" <<'PY'
import json, sys
from pathlib import Path
print("hybrid_h3_h8 = H9 (H7 checkpoint, limit 0.25, margin 0.25)")
print(f"{'agent':24s} {'decisions':>10s} {'override':>9s} {'|dQ|max':>9s}")
for path in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    n = len(rows)
    res = sum(row["residual_override"] for row in rows) / n
    peak = sum(row["residual_max_abs"] for row in rows) / n
    print(f"{path.stem:24s} {n:10d} {res:9.4f} {peak:9.4f}")
PY
