#!/usr/bin/env bash
# H3: first residual Double DQN with the default settings, seed 1401; then 4 x 25 rounds against
# the q-agent.

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_ROUNDS="${TRAIN_ROUNDS:-2000}"
EVAL_ROUNDS="${EVAL_ROUNDS:-25}"
SEED="${SEED:-1401}"
OUT="$PWD/results/q_agent_hybrid/hybrid"
RUN="$OUT/runs/H3_residual_ddqn_seed${SEED}"
CHECKPOINT="$OUT/checkpoints/H3_residual_ddqn_seed${SEED}.pt"
mkdir -p "$RUN" "$(dirname "$CHECKPOINT")"
rm -f "$CHECKPOINT" "$RUN/metrics.jsonl"

CUDA_VISIBLE_DEVICES='' BOMBERMAN_SEED="$SEED" \
  BOMBERMAN_Q_AGENT_HYBRID_MODEL="$CHECKPOINT" \
  BOMBERMAN_Q_AGENT_HYBRID_METRICS="$RUN/metrics.jsonl" \
  .MLE/bin/python main.py play --no-gui \
    --agents hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent q_agent/q_agent_v1 \
    --train 1 --scenario classic --seed "$SEED" --n-rounds "$TRAIN_ROUNDS" \
    --save-stats "$RUN/train_stats.json"

for seat in 0 1 2 3; do
  case "$seat" in
    0) roster="hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent q_agent/q_agent_v1" ;;
    1) roster="q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent" ;;
    2) roster="rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8 rule_based_agent" ;;
    3) roster="rule_based_agent rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8" ;;
  esac
  CUDA_VISIBLE_DEVICES='' BOMBERMAN_SEED="$SEED" \
    BOMBERMAN_Q_AGENT_HYBRID_MODEL="$CHECKPOINT" \
    .MLE/bin/python main.py play --no-gui --agents $roster --train 0 \
      --scenario classic --seed "$((SEED + 100 + seat))" --n-rounds "$EVAL_ROUNDS" \
      --save-stats "$RUN/eval_seat${seat}.json"
done
