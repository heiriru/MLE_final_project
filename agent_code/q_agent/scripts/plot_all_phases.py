#!/usr/bin/env python
"""
Plot ALL training phases of an agent as one continuous curve.

Reads every stats_*.json in the agent folder (saved per phase by
train_curriculum.sh), stitches them in training order (by file mtime) and
draws vertical lines + labels at the phase boundaries.

Usage (from the repo root):
    python scripts/plot_all_phases.py                 # default: pbs_nstep_agent
    python scripts/plot_all_phases.py qtable_agent
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

AGENT_DIR = Path(__file__).resolve().parent.parent


def moving_average(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main(agent_name):
    d = AGENT_DIR / agent_name
    files = sorted(d.glob("stats_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"No stats_*.json in {d}. Run the curriculum first.")

    metrics = ["reward", "coins", "steps", "invalid", "kills", "epsilon"]
    merged = {m: [] for m in metrics}
    bounds, labels = [], []          # phase boundaries (x) and names
    x = 0
    for f in files:
        s = json.loads(f.read_text())
        n = len(s.get("round", []))
        if n == 0:
            continue
        for m in metrics:
            merged[m].extend(s.get(m, [np.nan] * n))
        x += n
        bounds.append(x)
        labels.append(f.stem.replace("stats_", ""))

    metrics = [m for m in metrics if any(np.isfinite(merged[m]))]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 2.0 * len(metrics)),
                             sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    xs = np.arange(len(merged[metrics[0]]))
    for ax, m in zip(axes, metrics):
        y = np.array(merged[m], dtype=float)
        ax.plot(xs, y, alpha=0.25, lw=0.6)
        ma = moving_average(y)
        ax.plot(xs[-len(ma):], ma, lw=1.6)
        ax.set_ylabel(m)
        ax.grid(alpha=0.3)
        for b in bounds[:-1]:
            ax.axvline(b, color="k", ls="--", lw=0.7, alpha=0.5)
    # phase labels along the top axis
    start = 0
    for b, lab in zip(bounds, labels):
        axes[0].text((start + b) / 2, axes[0].get_ylim()[1], lab,
                     ha="center", va="bottom", fontsize=7, rotation=0)
        start = b
    axes[-1].set_xlabel("cumulative round (all phases)")
    fig.suptitle(f"Training across all phases: {agent_name}")
    fig.tight_layout()

    out = d / "training_all_phases.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pbs_nstep_agent")
