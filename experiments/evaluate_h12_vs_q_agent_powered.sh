#!/usr/bin/env bash
# Powered comparison: 8 seed bases x 4 seats x 200 rounds = 6,400 rounds of H12 vs the q-agent.

set -euo pipefail
cd "$(dirname "$0")/.."
EVAL_ROUNDS="${EVAL_ROUNDS:-200}"
BASES="${BASES:-6001 6101 6201 6301 6401 6501 6601 6701}"
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/H12_vs_q_agent_powered"
CK="$PWD/results/q_agent_hybrid/hybrid/checkpoints"
[[ -f "$CK/H12_authority_seed1501.pt" ]] || { echo "missing H12 checkpoint" >&2; exit 1; }
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1 Q_AGENT_H12_INPUT_DIM=14
export BOMBERMAN_Q_AGENT_H12_MODEL="$CK/H12_authority_seed1501.pt"

for base in $BASES; do
  for seat in 0 1 2 3; do
    tag="base${base}_seat${seat}"
    case "$seat" in
      0) roster="hybrid_agent/hybrid_h12 rule_based_agent rule_based_agent q_agent/q_agent_v1";;
      1) roster="q_agent/q_agent_v1 hybrid_agent/hybrid_h12 rule_based_agent rule_based_agent";;
      2) roster="rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h12 rule_based_agent";;
      3) roster="rule_based_agent rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h12";;
    esac
    log_dir="$OUT/logs_$tag"
    rm -rf "$log_dir"; mkdir -p "$log_dir"
    BOMBERMAN_SEED="$base" .MLE/bin/python main.py play --no-gui \
      --agents $roster --train 0 --scenario classic --log-dir "$log_dir" \
      --seed "$((base + 10 * seat))" --n-rounds "$EVAL_ROUNDS" \
      --save-stats "$OUT/$tag.json"
    .MLE/bin/python experiments/extract_round_points.py "$log_dir/game.log" "$OUT/$tag.rounds.json"
    rm -rf "$log_dir"
  done
  echo "=== base $base done"
done
echo POWERED_DONE
