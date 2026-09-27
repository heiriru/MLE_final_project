# H8a: hill climbing on the network's output layer, directly on game score.
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from agent_code.hybrid_agent.hybrid_h3_h8.agent import ResidualQNetwork

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/q_agent_hybrid/hybrid/runs/H8_hillclimb_seed1701"
CHECKPOINTS = OUT / "checkpoints"
ROSTERS = (
    "hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent q_agent/q_agent_v1",
    "q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent",
    "rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8 rule_based_agent",
    "rule_based_agent rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8",
)


# Saves a network as a checkpoint file.
def save(model: ResidualQNetwork, path: Path, iteration: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"residual": model.state_dict(), "iteration": iteration,
                "config": {"method": "paired_head_hillclimb", "device": "cpu"}}, path)


# Plays a few rounds with the given checkpoint and returns its score per round.
def score(checkpoint: Path, seed: int, seat: int, rounds: int, tag: str) -> float:
    stats = OUT / f"{tag}.json"
    env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "Q_AGENT_H3_CPU_THREADS": "1",
                        "Q_AGENT_RESIDUAL_LIMIT": "0.25", "Q_AGENT_OVERRIDE_MARGIN": "0.05",
                        "BOMBERMAN_Q_AGENT_HYBRID_MODEL": str(checkpoint)}
    subprocess.run([str(ROOT / ".MLE/bin/python"), "main.py", "play", "--no-gui", "--agents",
                    *ROSTERS[seat].split(), "--train", "0", "--scenario", "classic", "--seed", str(seed),
                    "--n-rounds", str(rounds), "--save-stats", str(stats)], cwd=ROOT, env=env, check=True)
    return float(json.loads(stats.read_text())["by_agent"]["hybrid_h3_h8"]["score"]) / rounds


# Hill climbing: add small random noise to the output layer and keep the candidate if it scores
# more than the current network on the same seed.
def main() -> None:
    iterations, rounds, sigma = int(os.environ.get("HILLCLIMB_ITERATIONS", "10")), int(os.environ.get("HILLCLIMB_ROUNDS", "10")), float(os.environ.get("HILLCLIMB_SIGMA", "0.02"))
    OUT.mkdir(parents=True, exist_ok=True)
    rng, incumbent = np.random.default_rng(1701), ResidualQNetwork()
    incumbent_path = CHECKPOINTS / "incumbent.pt"
    save(incumbent, incumbent_path, 0)
    with (OUT / "acceptance.jsonl").open("w", encoding="utf-8") as log:
        for iteration in range(iterations):
            candidate = ResidualQNetwork(); candidate.load_state_dict(incumbent.state_dict())

            with torch.no_grad():
                candidate.output.weight.add_(torch.as_tensor(rng.normal(0, sigma, candidate.output.weight.shape), dtype=torch.float32))
                candidate.output.bias.add_(torch.as_tensor(rng.normal(0, sigma, candidate.output.bias.shape), dtype=torch.float32))
            candidate_path = CHECKPOINTS / f"candidate_{iteration:03d}.pt"
            save(candidate, candidate_path, iteration + 1)
            seed, seat = 1701 + iteration, iteration % len(ROSTERS)
            before = score(incumbent_path, seed, seat, rounds, f"incumbent_{iteration:03d}")
            after = score(candidate_path, seed, seat, rounds, f"candidate_{iteration:03d}")
            accepted = after > before
            if accepted:
                incumbent = candidate
                save(incumbent, incumbent_path, iteration + 1)
            log.write(json.dumps({"iteration": iteration, "seed": seed, "seat": seat, "rounds": rounds,
                                  "incumbent_score_per_round": before, "candidate_score_per_round": after,
                                  "sigma": sigma, "accepted": accepted}) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
