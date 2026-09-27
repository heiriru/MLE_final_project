#!/usr/bin/env python
"""
Plot the training curves saved by an agent's train.py.

Usage (from the repo root):
    python scripts/plot_training.py linear_agent
    python scripts/plot_training.py dqn_agent

Produces a PNG next to the stats file and shows it. Use these curves in the
"Experiments and Results" section of your report -- systematic evaluation of
how each change affects performance is the single most important grading
criterion.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

AGENT_DIR = Path(__file__).resolve().parent.parent


def moving_average(x, w=25):
    if len(x) < w:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main(agent_name):
    stats_path = AGENT_DIR / agent_name / "training_stats.json"
    if not stats_path.exists():
        sys.exit(f"No stats found at {stats_path}. Train the agent first.")
    stats = json.loads(stats_path.read_text())

    metrics = [m for m in ["reward", "coins", "steps", "invalid", "kills",
                           "loss", "epsilon"] if m in stats and any(stats[m])]
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    rounds = stats["round"]
    for ax, m in zip(axes, metrics):
        ax.plot(rounds, stats[m], alpha=0.3, label="per round")
        ma = moving_average(stats[m])
        ax.plot(rounds[-len(ma):], ma, label="moving avg")
        ax.set_ylabel(m)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("round")
    fig.suptitle(f"Training progress: {agent_name}")
    fig.tight_layout()

    out = AGENT_DIR / agent_name / "training_progress.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "linear_agent")
