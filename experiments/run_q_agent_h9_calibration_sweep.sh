#!/usr/bin/env bash
# H9: 3 x 3 sweep of limit and margin on the H7 network, no training; one 200-round block per
# cell.

set -euo pipefail
cd "$(dirname "$0")/.."
OUT="$PWD/results/q_agent_hybrid/hybrid/runs/H9_calibration_sweep_200"
mkdir -p "$OUT"
LOG="$OUT/results.jsonl"
: > "$LOG"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export Q_AGENT_H3_CPU_THREADS=1 Q_AGENT_HYBRID_INPUT_DIM=6 BOMBERMAN_SEED=2201
export Q_AGENT_RESIDUAL_LIMIT=0.25 Q_AGENT_OVERRIDE_MARGIN=0.25
export BOMBERMAN_Q_AGENT_H9_MODEL="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H7_six_inputs_seed1501.pt"
export BOMBERMAN_Q_AGENT_HYBRID_MODEL="$PWD/results/q_agent_hybrid/hybrid/checkpoints/H7_six_inputs_seed1501.pt"
for limit in 0.15 0.25 0.35; do
  for margin in 0.10 0.25 0.40; do
    tag="limit${limit}_margin${margin}"
    stats="$OUT/${tag}.json"
    H9_RESIDUAL_LIMIT="$limit" H9_OVERRIDE_MARGIN="$margin" \
      .MLE/bin/python main.py play --no-gui --agents hybrid_agent/hybrid_h9 hybrid_agent/hybrid_h3_h8 q_agent/q_agent_v1 rule_based_agent \
      --train 0 --scenario classic --seed 2201 --n-rounds 200 --save-stats "$stats"
    SWEEP_STATS="$stats" SWEEP_LOG="$LOG" SWEEP_LIMIT="$limit" SWEEP_MARGIN="$margin" .MLE/bin/python -c '
import json, os
d=json.load(open(os.environ["SWEEP_STATS"]))["by_agent"]
row={"residual_limit":float(os.environ["SWEEP_LIMIT"]),"override_margin":float(os.environ["SWEEP_MARGIN"])}
for name, values in d.items():
    rounds=values["rounds"]
    row[name]={key: (values.get(key,0)/rounds if key in ("score","coins","kills") else values.get(key,0)) for key in ("score","coins","kills","suicides","invalid","crates")}
with open(os.environ["SWEEP_LOG"], "a") as handle: handle.write(json.dumps(row)+"\n")
'
  done
done
