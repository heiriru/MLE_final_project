#!/usr/bin/env bash
# Common ladder field: each stage against 2 rule-based agents and the q-agent, seed 4801, 4 seats
# x 200 rounds. H10/H11 come from their own scripts (block A).

set -euo pipefail
cd "$(dirname "$0")/.."
EVAL_ROUNDS="${EVAL_ROUNDS:-200}"; SEED="${SEED:-4801}"
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/H9_H12_ladder_${EVAL_ROUNDS}"
CK="$PWD/results/q_agent_hybrid/hybrid/checkpoints"
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1
export Q_AGENT_H12_INPUT_DIM=14
export BOMBERMAN_Q_AGENT_H9_MODEL="$CK/H7_six_inputs_seed1501.pt"
export BOMBERMAN_Q_AGENT_H12_MODEL="$CK/H12_authority_seed1501.pt"

# Plays all four seat rotations for one stage in the ladder field.
run_arm () {
  for seat in 0 1 2 3; do
    case "$seat" in
      0) roster="$2 rule_based_agent rule_based_agent q_agent/q_agent_v1";;
      1) roster="q_agent/q_agent_v1 $2 rule_based_agent rule_based_agent";;
      2) roster="rule_based_agent q_agent/q_agent_v1 $2 rule_based_agent";;
      3) roster="rule_based_agent rule_based_agent q_agent/q_agent_v1 $2";;
    esac
    BOMBERMAN_SEED="$SEED" .MLE/bin/python main.py play --no-gui \
      --agents $roster --train 0 --scenario classic --log-dir "$OUT/logs" \
      --seed "$((SEED + 10 * seat))" --n-rounds "$EVAL_ROUNDS" \
      --save-stats "$OUT/$1_seat$seat.json"
  done
  echo "=== ladder arm $1 done"
}

for arm in ${ARMS:-h12 h9}; do
  case "$arm" in
    h12) run_arm h12 hybrid_agent/hybrid_h12;;
    h9)  run_arm h9 hybrid_agent/hybrid_h9;;
  esac
done
echo LADDER_DONE
