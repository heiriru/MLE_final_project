#!/usr/bin/env bash
# H11: 14 network inputs (the 8 extra observations added).

set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_ROUNDS="${TRAIN_ROUNDS:-2000}"; EVAL_ROUNDS="${EVAL_ROUNDS:-25}"; SEED=1501
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/H11_features_seed1501"
CHECKPOINT="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H11_features_seed1501.pt"
mkdir -p "$OUT/logs" "$(dirname "$CHECKPOINT")"; rm -f "$CHECKPOINT" "$OUT/metrics.jsonl"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1 Q_AGENT_H4_N_STEP=3 Q_AGENT_H11_INPUT_DIM=14
export Q_AGENT_H3_EPS_START=0.05 Q_AGENT_H3_EPS_FINAL=0.01 Q_AGENT_H3_EPS_DECAY=60000
export Q_AGENT_H3_LEARNING_STARTS=10000 Q_AGENT_H3_TRAIN_EVERY=4 Q_AGENT_H3_TARGET_EVERY=2000

BOMBERMAN_SEED="$SEED" BOMBERMAN_Q_AGENT_H11_MODEL="$CHECKPOINT" \
  BOMBERMAN_Q_AGENT_H11_METRICS="$OUT/metrics.jsonl" \
  BOMBERMAN_Q_AGENT_TRAIN_HISTORY="$OUT/history.jsonl" \
  .MLE/bin/python main.py play --no-gui \
    --agents hybrid_agent/hybrid_h11 rule_based_agent rule_based_agent q_agent/q_agent_v1 \
    --train 1 --scenario classic --log-dir "$OUT/logs" --seed "$SEED" \
    --n-rounds "$TRAIN_ROUNDS" --save-stats "$OUT/train_stats.json"
