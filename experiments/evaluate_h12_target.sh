#!/usr/bin/env bash
# H12, H9, the q-agent and a rule-based agent in the same games, 4 seats x 200 rounds.

set -euo pipefail
cd "$(dirname "$0")/.."
EVAL_ROUNDS="${EVAL_ROUNDS:-200}"; SEED="${SEED:-4801}"
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/H12_eval_${EVAL_ROUNDS}"
H12_CHECKPOINT="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H12_authority_seed1501.pt"
H9_CHECKPOINT="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H7_six_inputs_seed1501.pt"
for file in "$H12_CHECKPOINT" "$H9_CHECKPOINT"; do
  [[ -f "$file" ]] || { echo "missing checkpoint: $file" >&2; exit 1; }
done
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1 Q_AGENT_H12_INPUT_DIM=14
export BOMBERMAN_Q_AGENT_H12_MODEL="$H12_CHECKPOINT" BOMBERMAN_Q_AGENT_H9_MODEL="$H9_CHECKPOINT"

for seat in 0 1 2 3; do
  case "$seat" in
    0) roster="hybrid_agent/hybrid_h12 hybrid_agent/hybrid_h9 q_agent/q_agent_v1 rule_based_agent";;
    1) roster="hybrid_agent/hybrid_h9 q_agent/q_agent_v1 rule_based_agent hybrid_agent/hybrid_h12";;
    2) roster="q_agent/q_agent_v1 rule_based_agent hybrid_agent/hybrid_h12 hybrid_agent/hybrid_h9";;
    3) roster="rule_based_agent hybrid_agent/hybrid_h12 hybrid_agent/hybrid_h9 q_agent/q_agent_v1";;
  esac
  BOMBERMAN_SEED="$SEED" .MLE/bin/python main.py play --no-gui \
    --agents $roster --train 0 --scenario classic --log-dir "$OUT/logs" \
    --seed "$((SEED + 10 * seat))" --n-rounds "$EVAL_ROUNDS" \
    --save-stats "$OUT/target_seat$seat.json"
done
