#!/usr/bin/env bash
# Without the q-agent in the game: H12 and the q-agent each against 3 rule-based agents on the
# same maps; plus the q-agent control on the replication maps (seed 8201).
set -euo pipefail
cd "$(dirname "$0")/.."

EVAL_ROUNDS="${EVAL_ROUNDS:-200}"
EVAL_SEED="${EVAL_SEED:-5801}"
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/H12_heldout_rules_${EVAL_ROUNDS}"
CONTROL="$PWD/results/q_agent_hybrid/hybrid/runs/H12_replication_q_agent_control"
CHECKPOINT="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H12_authority_seed1501.pt"
[[ -f "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 1; }
mkdir -p "$OUT/logs" "$CONTROL/logs"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1 Q_AGENT_H12_INPUT_DIM=14 BOMBERMAN_Q_AGENT_H12_MODEL="$CHECKPOINT"

for seat in 0 1 2 3; do
  case "$seat" in
    0) h12="hybrid_agent/hybrid_h12 rule_based_agent rule_based_agent rule_based_agent"; q_agent="q_agent/q_agent_v1 rule_based_agent rule_based_agent rule_based_agent";;
    1) h12="rule_based_agent hybrid_agent/hybrid_h12 rule_based_agent rule_based_agent"; q_agent="rule_based_agent q_agent/q_agent_v1 rule_based_agent rule_based_agent";;
    2) h12="rule_based_agent rule_based_agent hybrid_agent/hybrid_h12 rule_based_agent"; q_agent="rule_based_agent rule_based_agent q_agent/q_agent_v1 rule_based_agent";;
    3) h12="rule_based_agent rule_based_agent rule_based_agent hybrid_agent/hybrid_h12"; q_agent="rule_based_agent rule_based_agent rule_based_agent q_agent/q_agent_v1";;
  esac
  BOMBERMAN_SEED="$EVAL_SEED" .MLE/bin/python main.py play --no-gui \
    --agents $h12 --train 0 --scenario classic --log-dir "$OUT/logs" \
    --seed "$((EVAL_SEED + 10 * seat))" --n-rounds "$EVAL_ROUNDS" --save-stats "$OUT/h12_seat${seat}.json"
  BOMBERMAN_SEED="$EVAL_SEED" .MLE/bin/python main.py play --no-gui \
    --agents $q_agent --train 0 --scenario classic --log-dir "$CONTROL/logs" \
    --seed "$((EVAL_SEED + 10 * seat))" --n-rounds "$EVAL_ROUNDS" --save-stats "$OUT/q_agent_seat${seat}.json"
  if [[ "$EVAL_ROUNDS" == 100 ]]; then cp "$OUT/q_agent_seat${seat}.json" "$CONTROL/heldout_rules_seat${seat}.json"; fi
done

CONTROL_SEED=8201
for seat in 0 1 2 3; do
  case "$seat" in
    0) q_agent="q_agent/q_agent_v1 rule_based_agent rule_based_agent rule_based_agent";;
    1) q_agent="rule_based_agent q_agent/q_agent_v1 rule_based_agent rule_based_agent";;
    2) q_agent="rule_based_agent rule_based_agent q_agent/q_agent_v1 rule_based_agent";;
    3) q_agent="rule_based_agent rule_based_agent rule_based_agent q_agent/q_agent_v1";;
  esac
  BOMBERMAN_SEED="$CONTROL_SEED" .MLE/bin/python main.py play --no-gui \
    --agents $q_agent --train 0 --scenario classic --log-dir "$CONTROL/logs" \
    --seed "$((CONTROL_SEED + 10 * seat))" --n-rounds 100 \
    --save-stats "$CONTROL/heldout_rules_seat${seat}.json"
done
