# H8b: evolution strategy with Adam on the output layer, directly on game score.
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from agent_code.hybrid_agent.hybrid_h3_h8.agent import ResidualQNetwork

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/q_agent_hybrid/hybrid/runs/H8_adam_es_seed1801"
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
        size = model.output.weight.numel()
        model.output.weight.copy_(vector[:size].view_as(model.output.weight))
        model.output.bias.copy_(vector[size:].view_as(model.output.bias))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"residual": model.state_dict(), "epoch": epoch,
                "config": {"method": "one_sample_adam_es", "objective": "minimize_negative_score_per_round",
                           "rounds_per_epoch": 5, "device": "cpu"}}, path)


# Plays 5 rounds with that checkpoint (the seat rotates per epoch) and returns the score per
# round.
def evaluate(checkpoint: Path, epoch: int, rounds: int) -> float:
    seat, seed = epoch % len(ROSTERS), 1801 + epoch
    stats = OUT / f"epoch_{epoch:04d}.json"
    env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "Q_AGENT_H3_CPU_THREADS": "1",
                        "Q_AGENT_RESIDUAL_LIMIT": "0.25", "Q_AGENT_OVERRIDE_MARGIN": "0.05",
                        "BOMBERMAN_Q_AGENT_HYBRID_MODEL": str(checkpoint)}
    subprocess.run([str(ROOT / ".MLE/bin/python"), "main.py", "play", "--no-gui", "--agents",
                    *ROSTERS[seat].split(), "--train", "0", "--scenario", "classic", "--seed", str(seed),
                    "--n-rounds", str(rounds), "--save-stats", str(stats)], cwd=ROOT, env=env, check=True)
    score = json.loads(stats.read_text())["by_agent"]["hybrid_h3_h8"]["score"]
    return float(score) / rounds


# Evolution strategy: perturb the output layer, estimate the score gradient from the result,
# update with Adam, 1,000 epochs.
def main() -> None:
    epochs = int(os.environ.get("ADAM_ES_EPOCHS", "1000"))
    rounds = int(os.environ.get("ADAM_ES_ROUNDS", "5"))
    sigma = float(os.environ.get("ADAM_ES_SIGMA", "0.02"))
    lr = float(os.environ.get("ADAM_ES_LR", "0.002"))
    baseline_decay = float(os.environ.get("ADAM_ES_BASELINE_DECAY", "0.95"))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1801)
    base = ResidualQNetwork()
    initial = torch.cat((base.output.weight.detach().flatten(), base.output.bias.detach().flatten()))
    theta = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([theta], lr=lr)
    rng, baseline = np.random.default_rng(1801), None
    best_score, best_vector = -float("inf"), theta.detach().clone()
    with (OUT / "epochs.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(epochs):
            noise = torch.as_tensor(rng.normal(size=theta.numel()), dtype=torch.float32)
            candidate = theta.detach() + sigma * noise
            candidate_path = CHECKPOINTS / "candidate.pt"
            save_head(base, candidate, candidate_path, epoch)
            score = evaluate(candidate_path, epoch, rounds)
            previous_baseline = score if baseline is None else baseline
            advantage = score - previous_baseline

            optimizer.zero_grad(set_to_none=True)
            theta.grad = -(advantage / sigma) * noise
            torch.nn.utils.clip_grad_norm_([theta], 10.0)
            optimizer.step()
            with torch.no_grad():
                theta.clamp_(-0.5, 0.5)
            baseline = score if baseline is None else baseline_decay * baseline + (1 - baseline_decay) * score
            if score > best_score:
                best_score, best_vector = score, candidate.clone()
                save_head(base, best_vector, CHECKPOINTS / "best.pt", epoch)
            save_head(base, theta.detach(), CHECKPOINTS / "mean.pt", epoch)
            log.write(json.dumps({"epoch": epoch, "seed": 1801 + epoch, "seat": epoch % 4,
                                  "score_per_round": score, "negative_score_objective": -score,
                                  "baseline": previous_baseline, "advantage": advantage,
                                  "sigma": sigma, "learning_rate": lr, "best_score_per_round": best_score}) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
