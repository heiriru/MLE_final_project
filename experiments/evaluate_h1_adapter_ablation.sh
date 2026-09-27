#!/usr/bin/env bash
# H1: which part of the q-agent matters - full table, no table, or WAIT for unseen states
# (25-round screens).

set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS="${ROUNDS:-25}"
SEED="${SEED:-909}"
OUTPUT="results/q_agent_hybrid/hybrid/runs/H1_adapter_ablation"
mkdir -p "$OUTPUT"

for mode in full no_q_table unseen_wait; do
  Q_AGENT_ADAPTER_MODE="$mode" .MLE/bin/python main.py play --no-gui \
    --agents hybrid_agent/adapter rule_based_agent rule_based_agent rule_based_agent \
    --train 0 --scenario classic --seed "$SEED" --n-rounds "$ROUNDS" \
    --save-stats "$OUTPUT/${mode}_classic_${ROUNDS}.json"
done
