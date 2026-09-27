#!/usr/bin/env bash
# H0: the q-agent against an identical copy of itself - how large are differences by pure chance?

set -euo pipefail
cd "$(dirname "$0")/.."
EVAL_ROUNDS="${EVAL_ROUNDS:-200}"; SEED="${SEED:-4801}"
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/H0_noise_floor_${EVAL_ROUNDS}"
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1

for seat in 0 1 2 3; do
  case "$seat" in
    0) roster="q_agent/q_agent_v1 rule_based_agent rule_based_agent q_agent/q_agent_v1_copy";;
    1) roster="q_agent/q_agent_v1_copy q_agent/q_agent_v1 rule_based_agent rule_based_agent";;
    2) roster="rule_based_agent q_agent/q_agent_v1_copy q_agent/q_agent_v1 rule_based_agent";;
    3) roster="rule_based_agent rule_based_agent q_agent/q_agent_v1_copy q_agent/q_agent_v1";;
  esac
  BOMBERMAN_SEED="$SEED" .MLE/bin/python main.py play --no-gui \
    --agents $roster --train 0 --scenario classic --log-dir "$OUT/logs" \
    --seed "$((SEED + 10 * seat))" --n-rounds "$EVAL_ROUNDS" \
    --save-stats "$OUT/null_seat$seat.json"
done
echo D0_NULL_DONE
