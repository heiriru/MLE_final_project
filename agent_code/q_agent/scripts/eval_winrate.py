#!/usr/bin/env python3
"""
Spielt N Spiele (je 1 Runde) mit dem Agenten gegen die Gegner und wertet aus:
  - wie oft der Agent gewinnt (hoechster Score), unentschieden, verliert
  - Verteilung der erreichten Punkte
Erzeugt einen sauberen matplotlib-Plot + Konsolen-Zusammenfassung.

Benutzung (im Projekt-Root oder ueberall):
    conda activate bomberman
    python scripts/eval_winrate.py            # 100 Spiele (Default)
    python scripts/eval_winrate.py 50         # 50 Spiele
    python scripts/eval_winrate.py 100 classic peaceful_agent peaceful_agent peaceful_agent
Argumente: [N_SPIELE] [SZENARIO] [GEGNER ...]
"""
import os, sys, json, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg") if os.environ.get("MPLBACKEND") == "Agg" else None
import matplotlib.pyplot as plt

# ---------------- Konfiguration ----------------
AGENT     = "q_agent/BOMBARIO"
FRAMEWORK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

N_GAMES   = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SCENARIO  = sys.argv[2] if len(sys.argv) > 2 else "classic"
OPPONENTS = sys.argv[3:] if len(sys.argv) > 3 else ["rule_based_agent"] * 3

# Farben (colorblind-freundlich; Identitaet zusaetzlich ueber Achsentexte)
C_WIN, C_TIE, C_LOSS = "#4E79A7", "#BAB0AC", "#E1575A"
C_HIST, C_MEAN       = "#4E79A7", "#E1575A"
INK, MUTED, GRID     = "#2b2b2b", "#6b6b6b", "#d9d9d9"

def run_game(stats_path):
    """Spielt genau eine Runde, liefert dict {agent_name: score} oder None."""
    cmd = [sys.executable, "main.py", "play",
           "--agents", AGENT, *OPPONENTS,
           "--no-gui", "--n-rounds", "1", "--save-stats", stats_path]
    try:
        r = subprocess.run(cmd, cwd=FRAMEWORK, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not os.path.isfile(stats_path):
        return None
    with open(stats_path) as f:
        d = json.load(f)
    return {name: info.get("score", 0) for name, info in d["by_agent"].items()}

def main():
    print(f"Spiele {N_GAMES}x '{SCENARIO}':  {AGENT}  vs  {', '.join(OPPONENTS)}")
    my_scores, opp_best, outcomes = [], [], {"win": 0, "tie": 0, "loss": 0}
    errors = 0
    tmpdir = tempfile.mkdtemp(prefix="winrate_")
    for i in range(N_GAMES):
        scores = run_game(os.path.join(tmpdir, "s.json"))
        if scores is None:
            errors += 1
            continue
        my_key = next((k for k in scores if k == os.path.basename(AGENT) or k.startswith(os.path.basename(AGENT))), None)
        if my_key is None:
            errors += 1
            continue
        my = scores[my_key]
        best_other = max((v for k, v in scores.items() if k != my_key), default=0)
        my_scores.append(my)
        opp_best.append(best_other)
        if my > best_other:   outcomes["win"]  += 1
        elif my == best_other: outcomes["tie"] += 1
        else:                  outcomes["loss"] += 1
        if (i + 1) % 10 == 0 or i + 1 == N_GAMES:
            done = i + 1 - errors
            wr = 100 * outcomes["win"] / max(done, 1)
            print(f"  {i+1:3d}/{N_GAMES}  Winrate bisher: {wr:5.1f}%", flush=True)

    played = len(my_scores)
    if played == 0:
        print("Keine gueltigen Spiele -- Abbruch."); sys.exit(1)
    my_scores = np.array(my_scores)
    opp_best  = np.array(opp_best)
    margins   = my_scores - opp_best          # >0: gewonnen, <0: verloren
    wr   = 100 * outcomes["win"]  / played
    tr   = 100 * outcomes["tie"]  / played
    lr   = 100 * outcomes["loss"] / played
    mean, med, mx = my_scores.mean(), np.median(my_scores), my_scores.max()
    win_m  = margins[margins > 0].mean() if (margins > 0).any() else 0.0
    loss_m = margins[margins < 0].mean() if (margins < 0).any() else 0.0

    print("\n==== Ergebnis ====")
    print(f"  gespielt: {played} (Fehler/uebersprungen: {errors})")
    print(f"  Siege {outcomes['win']} ({wr:.1f}%) | Unentschieden {outcomes['tie']} ({tr:.1f}%) | Niederlagen {outcomes['loss']} ({lr:.1f}%)")
    print(f"  Punkte: Ø {mean:.2f} | Median {med:.0f} | Max {mx:.0f}")
    print(f"  Punktedifferenz zum besten Gegner: Ø {margins.mean():+.2f}")
    print(f"    -> Ø Siegabstand {win_m:+.2f} | Ø Niederlagenabstand {loss_m:+.2f}")

    # ---------------- Plot ----------------
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.6))
    fig.suptitle(f"{AGENT}  –  {played} Spiele vs {', '.join(OPPONENTS)}  ({SCENARIO})",
                 fontsize=13, fontweight="bold", color=INK)

    # (A) Ergebnis-Balken
    labels = ["Siege", "Unentsch.", "Niederl."]
    vals   = [outcomes["win"], outcomes["tie"], outcomes["loss"]]
    cols   = [C_WIN, C_TIE, C_LOSS]
    bars = ax1.bar(labels, vals, color=cols, width=0.62, zorder=3)
    ax1.set_title(f"Gewinnrate: {wr:.1f}%", fontsize=12, color=INK, pad=8)
    ax1.set_ylabel("Anzahl Spiele")
    ax1.set_ylim(0, max(vals) * 1.18 + 1)
    ax1.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for sp in ("top", "right"): ax1.spines[sp].set_visible(False)
    for b, v in zip(bars, vals):
        pct = 100 * v / played
        ax1.text(b.get_x() + b.get_width()/2, v + max(vals)*0.02,
                 f"{v}\n{pct:.0f}%", ha="center", va="bottom", fontsize=10, color=INK)

    # (B) Punkte-Histogramm
    hi = int(my_scores.max())
    bins = np.arange(-0.5, hi + 1.5, 1)
    ax2.hist(my_scores, bins=bins, color=C_HIST, edgecolor="white", linewidth=0.8, zorder=3)
    ax2.axvline(mean, color=C_MEAN, linewidth=2, zorder=4,
                label=f"Ø {mean:.2f}")
    ax2.set_title("Punkteverteilung", fontsize=12, color=INK, pad=8)
    ax2.set_xlabel("Punkte pro Spiel"); ax2.set_ylabel("Anzahl Spiele")
    ax2.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
    ax2.legend(frameon=False, loc="upper right")

    # (C) Punktedifferenz zum besten Gegner (Polaritaet: gewonnen/verloren)
    lo, hi = int(margins.min()), int(margins.max())
    edges = np.arange(lo - 0.5, hi + 1.5, 1)
    counts, _ = np.histogram(margins, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    mcols = [C_WIN if c > 0 else (C_LOSS if c < 0 else C_TIE) for c in centers]
    ax3.bar(centers, counts, width=0.9, color=mcols, edgecolor="white", linewidth=0.8, zorder=3)
    ax3.axvline(0, color=MUTED, linewidth=1.5, zorder=4)                       # neutraler Mittelpunkt
    ax3.axvline(margins.mean(), color=INK, linewidth=2, linestyle="--", zorder=5,
                label=f"Ø {margins.mean():+.2f}")
    ax3.set_title("Punktedifferenz zum besten Gegner", fontsize=12, color=INK, pad=8)
    ax3.set_xlabel("eigene Punkte − beste Gegnerpunkte"); ax3.set_ylabel("Anzahl Spiele")
    ax3.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax3.set_axisbelow(True)
    for sp in ("top", "right"): ax3.spines[sp].set_visible(False)
    ax3.legend(frameon=False, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(ROOT, "bombario_winrate.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot gespeichert: {out}")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
