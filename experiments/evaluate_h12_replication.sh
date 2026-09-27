#!/usr/bin/env bash
# Retrained H12 (seeds 1502/1503) in the training-like field and the held-out field.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_SEED="${TRAIN_SEED:?TRAIN_SEED is required}"
EVAL_ROUNDS="${EVAL_ROUNDS:-100}"
EVAL_SEED="${EVAL_SEED:-8201}"
TAG="H12_replication_seed${TRAIN_SEED}"
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/$TAG"
CHECKPOINT="$PWD/results/q_agent_hybrid/hybrid/checkpoints/$TAG.pt"
[[ -f "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 1; }
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1 Q_AGENT_H12_INPUT_DIM=14
export BOMBERMAN_Q_AGENT_H12_MODEL="$CHECKPOINT"

for seat in 0 1 2 3; do
  case "$seat" in
    0) mixed="hybrid_agent/hybrid_h12 q_agent/q_agent_v1 rule_based_agent rule_based_agent"; heldout="hybrid_agent/hybrid_h12 rule_based_agent rule_based_agent rule_based_agent";;
    1) mixed="rule_based_agent hybrid_agent/hybrid_h12 q_agent/q_agent_v1 rule_based_agent"; heldout="rule_based_agent hybrid_agent/hybrid_h12 rule_based_agent rule_based_agent";;
    2) mixed="rule_based_agent rule_based_agent hybrid_agent/hybrid_h12 q_agent/q_agent_v1"; heldout="rule_based_agent rule_based_agent hybrid_agent/hybrid_h12 rule_based_agent";;
    3) mixed="q_agent/q_agent_v1 rule_based_agent rule_based_agent hybrid_agent/hybrid_h12"; heldout="rule_based_agent rule_based_agent rule_based_agent hybrid_agent/hybrid_h12";;
  esac
  for arm in mixed heldout; do
    roster="$mixed"; output="$OUT/mixed_seat${seat}.json"
    if [[ "$arm" == heldout ]]; then roster="$heldout"; output="$OUT/heldout_rules_seat${seat}.json"; fi
    BOMBERMAN_SEED="$EVAL_SEED" .MLE/bin/python main.py play --no-gui \
      --agents $roster --train 0 --scenario classic --log-dir "$OUT/logs" \
      --seed "$((EVAL_SEED + 10 * seat))" --n-rounds "$EVAL_ROUNDS" --save-stats "$output"
  done
done
