#!/usr/bin/env bash
# H4: one arm of the one-step vs three-step comparison (N_STEP and SEED select the arm).
set -euo pipefail
cd "$(dirname "$0")/.."
N_STEP="${N_STEP:?}"; SEED="${SEED:?}"
TRAIN_ROUNDS="${TRAIN_ROUNDS:-2000}"; EVAL_ROUNDS="${EVAL_ROUNDS:-25}"
RUN="$PWD/results/q_agent_hybrid/hybrid/runs/H4_nstep_equal_budget/n${N_STEP}_seed${SEED}"
CHECKPOINT="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H4_n${N_STEP}_seed${SEED}.pt"
rm -rf "$RUN"; mkdir -p "$RUN/logs" "$(dirname "$CHECKPOINT")"; rm -f "$CHECKPOINT"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1 Q_AGENT_H4_N_STEP="$N_STEP" BOMBERMAN_SEED="$SEED"
export BOMBERMAN_Q_AGENT_HYBRID_MODEL="$CHECKPOINT"
BOMBERMAN_Q_AGENT_HYBRID_METRICS="$RUN/metrics.jsonl" .MLE/bin/python main.py play --no-gui \
  --agents hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent q_agent/q_agent_v1 \
  --train 1 --scenario classic --log-dir "$RUN/logs" --seed "$SEED" \
  --n-rounds "$TRAIN_ROUNDS" --save-stats "$RUN/train_stats.json"
for seat in 0 1 2 3; do
  case "$seat" in
    0) roster="hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent q_agent/q_agent_v1";;
    1) roster="q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent";;
    2) roster="rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8 rule_based_agent";;
    3) roster="rule_based_agent rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8";;
  esac
  .MLE/bin/python main.py play --no-gui --agents $roster --train 0 --scenario classic \
    --log-dir "$RUN/logs" --seed "$((SEED + 100 + seat))" --n-rounds "$EVAL_ROUNDS" \
    --save-stats "$RUN/eval_seat${seat}.json"
done
