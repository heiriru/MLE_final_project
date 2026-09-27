#!/usr/bin/env bash
# H10 evaluation: block A = ladder field, block B = direct games against H9.

set -euo pipefail
cd "$(dirname "$0")/.."
EVAL_ROUNDS="${EVAL_ROUNDS:-200}"; SEED="${SEED:-4801}"
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/H10_eval_${EVAL_ROUNDS}"
CHECKPOINT="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H10_dueling_seed1501.pt"
H9_CHECKPOINT="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H7_six_inputs_seed1501.pt"
[[ -f "$CHECKPOINT" ]] || { echo "missing H10 checkpoint: $CHECKPOINT" >&2; exit 1; }
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1 Q_AGENT_H10_INPUT_DIM=6
export BOMBERMAN_Q_AGENT_H10_MODEL="$CHECKPOINT" BOMBERMAN_Q_AGENT_H9_MODEL="$H9_CHECKPOINT"

# Plays one 200-round seat block with the given roster and seat seed.
run_seat () {
  BOMBERMAN_SEED="$SEED" .MLE/bin/python main.py play --no-gui \
    --agents $3 --train 0 --scenario classic --log-dir "$OUT/logs" \
    --seed "$((SEED + 10 * $2))" --n-rounds "$EVAL_ROUNDS" \
    --save-stats "$OUT/$1_seat$2.json"
}

for seat in 0 1 2 3; do
  case "$seat" in
    0) roster="hybrid_agent/hybrid_h10 rule_based_agent rule_based_agent q_agent/q_agent_v1";;
    1) roster="q_agent/q_agent_v1 hybrid_agent/hybrid_h10 rule_based_agent rule_based_agent";;
    2) roster="rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h10 rule_based_agent";;
    3) roster="rule_based_agent rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h10";;
  esac
  run_seat blockA "$seat" "$roster"
done

for seat in 0 1 2 3; do
  case "$seat" in
    0) roster="hybrid_agent/hybrid_h10 hybrid_agent/hybrid_h9 q_agent/q_agent_v1 rule_based_agent";;
    1) roster="hybrid_agent/hybrid_h9 q_agent/q_agent_v1 rule_based_agent hybrid_agent/hybrid_h10";;
    2) roster="q_agent/q_agent_v1 rule_based_agent hybrid_agent/hybrid_h10 hybrid_agent/hybrid_h9";;
    3) roster="rule_based_agent hybrid_agent/hybrid_h10 hybrid_agent/hybrid_h9 q_agent/q_agent_v1";;
  esac
  run_seat blockB "$seat" "$roster"
done
