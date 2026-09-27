#!/usr/bin/env bash
# H10: adds the value head V(s); everything else as in H7.

set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_ROUNDS="${TRAIN_ROUNDS:-2000}"; EVAL_ROUNDS="${EVAL_ROUNDS:-25}"; SEED=1501
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/H10_dueling_seed1501"
CHECKPOINT="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H10_dueling_seed1501.pt"
H9_CHECKPOINT="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H7_six_inputs_seed1501.pt"
mkdir -p "$OUT/logs" "$(dirname "$CHECKPOINT")"; rm -f "$CHECKPOINT" "$OUT/metrics.jsonl"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1 Q_AGENT_H4_N_STEP=3 Q_AGENT_H10_INPUT_DIM=6
export Q_AGENT_H3_EPS_START=0.05 Q_AGENT_H3_EPS_FINAL=0.01 Q_AGENT_H3_EPS_DECAY=60000
export Q_AGENT_H3_LEARNING_STARTS=10000 Q_AGENT_H3_TRAIN_EVERY=4 Q_AGENT_H3_TARGET_EVERY=2000
export BOMBERMAN_Q_AGENT_H9_MODEL="$H9_CHECKPOINT"

BOMBERMAN_SEED="$SEED" BOMBERMAN_Q_AGENT_H10_MODEL="$CHECKPOINT" \
  BOMBERMAN_Q_AGENT_H10_METRICS="$OUT/metrics.jsonl" \
  BOMBERMAN_Q_AGENT_TRAIN_HISTORY="$OUT/history.jsonl" \
  .MLE/bin/python main.py play --no-gui \
    --agents hybrid_agent/hybrid_h10 rule_based_agent rule_based_agent q_agent/q_agent_v1 \
    --train 1 --scenario classic --log-dir "$OUT/logs" --seed "$SEED" \
    --n-rounds "$TRAIN_ROUNDS" --save-stats "$OUT/train_stats.json"

for seat in 0 1 2 3; do
  case "$seat" in
    0) roster="hybrid_agent/hybrid_h10 rule_based_agent rule_based_agent q_agent/q_agent_v1";;
    1) roster="q_agent/q_agent_v1 hybrid_agent/hybrid_h10 rule_based_agent rule_based_agent";;
    2) roster="rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h10 rule_based_agent";;
    3) roster="rule_based_agent rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h10";;
  esac
  BOMBERMAN_SEED="$SEED" BOMBERMAN_Q_AGENT_H10_MODEL="$CHECKPOINT" \
    .MLE/bin/python main.py play --no-gui --agents $roster --train 0 \
      --scenario classic --log-dir "$OUT/logs" --seed "$((SEED + 100 + seat))" \
      --n-rounds "$EVAL_ROUNDS" --save-stats "$OUT/eval_seat${seat}.json"
done

for seat in 0 1; do
  case "$seat" in
    0) roster="hybrid_agent/hybrid_h10 rule_based_agent hybrid_agent/hybrid_h9 q_agent/q_agent_v1";;
    1) roster="hybrid_agent/hybrid_h9 q_agent/q_agent_v1 rule_based_agent hybrid_agent/hybrid_h10";;
  esac
  BOMBERMAN_SEED="$SEED" BOMBERMAN_Q_AGENT_H10_MODEL="$CHECKPOINT" \
    .MLE/bin/python main.py play --no-gui --agents $roster --train 0 \
      --scenario classic --log-dir "$OUT/logs" --seed "$((SEED + 200 + seat))" \
      --n-rounds "$EVAL_ROUNDS" --save-stats "$OUT/versus_h9_seat${seat}.json"
done
