# H8c: antithetic evolution strategy (+ and - perturbation on the same seeds) with Adam.
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from agent_code.hybrid_agent.hybrid_h3_h8.agent import ResidualQNetwork

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/q_agent_hybrid/hybrid/runs/H8_antithetic_es_seed1901"
CHECKPOINTS = OUT / "checkpoints"
ROSTERS = (
    "hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent q_agent/q_agent_v1",
    "q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8 rule_based_agent rule_based_agent",
    "rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8 rule_based_agent",
    "rule_based_agent rule_based_agent q_agent/q_agent_v1 hybrid_agent/hybrid_h3_h8",
)


# Writes a parameter vector into the output layer of a network and saves it as a checkpoint.
def save_head(base: ResidualQNetwork, vector: torch.Tensor, path: Path, epoch: int) -> None:
    model = ResidualQNetwork(); model.load_state_dict(base.state_dict())
    with torch.no_grad():
        count = model.output.weight.numel()
        model.output.weight.copy_(vector[:count].view_as(model.output.weight))
        model.output.bias.copy_(vector[count:].view_as(model.output.bias))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"residual": model.state_dict(), "epoch": epoch,
                "config": {"method": "antithetic_common_random_number_adam_es", "device": "cpu"}}, path)


# Plays 5 rounds with that checkpoint (the seat rotates per epoch) and returns the score per
# round.
def evaluate(checkpoint: Path, epoch: int, sign: str, rounds: int) -> float:
    seat, seed = epoch % len(ROSTERS), 1901 + epoch
    stats = OUT / f"epoch_{epoch:04d}_{sign}.json"
    env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "Q_AGENT_H3_CPU_THREADS": "1",
                        "Q_AGENT_RESIDUAL_LIMIT": "0.25", "Q_AGENT_OVERRIDE_MARGIN": "0.05",
                        "BOMBERMAN_Q_AGENT_HYBRID_MODEL": str(checkpoint)}
    subprocess.run([str(ROOT / ".MLE/bin/python"), "main.py", "play", "--no-gui", "--agents",
                    *ROSTERS[seat].split(), "--train", "0", "--scenario", "classic", "--seed", str(seed),
                    "--n-rounds", str(rounds), "--save-stats", str(stats)], cwd=ROOT, env=env, check=True)
    return float(json.loads(stats.read_text())["by_agent"]["hybrid_h3_h8"]["score"]) / rounds


# Antithetic ES: evaluate +noise and -noise on the same seed, gradient = score difference / (2
# sigma) * noise, Adam update; keeps the best vector and the running mean.
def main() -> None:
    epochs = int(os.environ.get("ANTITHETIC_ES_EPOCHS", "1000"))
    rounds = int(os.environ.get("ANTITHETIC_ES_ROUNDS", "5"))
    sigma = float(os.environ.get("ANTITHETIC_ES_SIGMA", "0.02"))
    lr = float(os.environ.get("ANTITHETIC_ES_LR", "0.001"))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1901)
    base = ResidualQNetwork()
    theta = torch.nn.Parameter(torch.cat((base.output.weight.detach().flatten(), base.output.bias.detach().flatten())))
    optimizer, rng = torch.optim.Adam([theta], lr=lr), np.random.default_rng(1901)
    best_score, best = -float("inf"), theta.detach().clone()
    with (OUT / "epochs.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(epochs):
            noise = torch.as_tensor(rng.normal(size=theta.numel()), dtype=torch.float32)
            plus, minus = theta.detach() + sigma * noise, theta.detach() - sigma * noise
            plus_path, minus_path = CHECKPOINTS / "plus.pt", CHECKPOINTS / "minus.pt"
            save_head(base, plus, plus_path, epoch); save_head(base, minus, minus_path, epoch)
            plus_score = evaluate(plus_path, epoch, "plus", rounds)
            minus_score = evaluate(minus_path, epoch, "minus", rounds)

            gradient_ascent = ((plus_score - minus_score) / (2 * sigma)) * noise
            optimizer.zero_grad(set_to_none=True)
            theta.grad = -gradient_ascent
            torch.nn.utils.clip_grad_norm_([theta], 10.0)
            optimizer.step()
            with torch.no_grad(): theta.clamp_(-0.5, 0.5)
            average = (plus_score + minus_score) / 2
            chosen, chosen_score = (plus, plus_score) if plus_score >= minus_score else (minus, minus_score)
            if chosen_score > best_score:
                best_score, best = chosen_score, chosen.clone()
                save_head(base, best, CHECKPOINTS / "best.pt", epoch)
            save_head(base, theta.detach(), CHECKPOINTS / "mean.pt", epoch)
            log.write(json.dumps({"epoch": epoch, "seed": 1901 + epoch, "seat": epoch % 4, "rounds_per_sign": rounds,
                                  "plus_score_per_round": plus_score, "minus_score_per_round": minus_score,
                                  "paired_difference": plus_score - minus_score, "pair_mean_score_per_round": average,
                                  "sigma": sigma, "learning_rate": lr, "best_score_per_round": best_score}) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
