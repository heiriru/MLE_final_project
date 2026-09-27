#!/usr/bin/env bash
# H6: final replay and training schedule (10,000 warm-up transitions, update every 4, target copy
# every 2,000, epsilon 0.05 -> 0.01).

set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_ROUNDS="${TRAIN_ROUNDS:-2000}"; EVAL_ROUNDS="${EVAL_ROUNDS:-25}"; SEED=1501
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/H6_replay_repair_seed1501"
CHECKPOINT="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H6_replay_repair_seed1501.pt"
mkdir -p "$OUT" "$(dirname "$CHECKPOINT")"; rm -f "$CHECKPOINT" "$OUT/metrics.jsonl"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1 Q_AGENT_H4_N_STEP=3
export Q_AGENT_H3_EPS_START=0.05 Q_AGENT_H3_EPS_FINAL=0.01 Q_AGENT_H3_EPS_DECAY=60000
export Q_AGENT_H3_LEARNING_STARTS=10000 Q_AGENT_H3_TRAIN_EVERY=4 Q_AGENT_H3_TARGET_EVERY=2000
BOMBERMAN_SEED="$SEED" BOMBERMAN_Q_AGENT_HYBRID_MODEL="$CHECKPOINT" BOMBERMAN_Q_AGENT_HYBRID_METRICS="$OUT/metrics.jsonl" .MLE/bin/python main.py play --no-gui --agents hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent q_agent/q_agent_v1 --train 1 --scenario classic --seed "$SEED" --n-rounds "$TRAIN_ROUNDS" --save-stats "$OUT/train_stats.json"
for seat in 0 1 2 3; do
  case "$seat" in
    0) roster="hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent q_agent/q_agent_v1";;
    1) roster="q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent";;
    2) roster="rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8 rule_based_agent";;
    3) roster="rule_based_agent rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8";;
  esac
  BOMBERMAN_SEED="$SEED" BOMBERMAN_Q_AGENT_HYBRID_MODEL="$CHECKPOINT" .MLE/bin/python main.py play --no-gui --agents $roster --train 0 --scenario classic --seed "$((SEED + 100 + seat))" --n-rounds "$EVAL_ROUNDS" --save-stats "$OUT/eval_seat${seat}.json"
done
