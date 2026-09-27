#!/usr/bin/env bash
# H8c evaluation: the best and the final (mean) ES vector, 4 x 25 rounds against the q-agent.

set -euo pipefail
cd "$(dirname "$0")/.."
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/H8_antithetic_es_seed1901"
SEED=1901
ROUNDS=25
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1 Q_AGENT_RESIDUAL_LIMIT=0.25 Q_AGENT_OVERRIDE_MARGIN=0.25
for selection in best mean; do
  checkpoint="$OUT/checkpoints/$selection.pt"
  for seat in 0 1 2 3; do
    case "$seat" in
      0) roster="hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent q_agent/q_agent_v1";;
      1) roster="q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent";;
      2) roster="rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8 rule_based_agent";;
      3) roster="rule_based_agent rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8";;
    esac
    BOMBERMAN_SEED="$SEED" BOMBERMAN_Q_AGENT_HYBRID_MODEL="$checkpoint" \
      .MLE/bin/python main.py play --no-gui --agents $roster --train 0 --scenario classic \
      --seed "$((SEED + 100 + seat))" --n-rounds "$ROUNDS" \
      --save-stats "$OUT/eval_${selection}_seat${seat}.json"
  done
done
