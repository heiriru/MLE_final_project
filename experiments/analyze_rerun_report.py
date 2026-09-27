# Computes every number of the hybrid part of the report from the raw result files ->
# REPORT_NUMBERS.md.
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/q_agent_hybrid/hybrid/runs"
Q_AGENT = "q_agent_v1"
RULE = "rule_based_agent"
T975 = {3: 3.182, 7: 2.365}


# Standard error of the mean
def se(values):
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")


# Our two-standard-error rule: 'resolved' if |mean| > 2 SE.
def verdict(delta, error):
    return "resolved" if abs(delta) > 2 * error else "unresolved"


# 'mean +- SE' as text.
def fmt(values):
    return f"{mean(values):.3f} +- {se(values):.3f}"


# Per-round value of one agent in one stats file (score = coins + 5*kills); copies of the same
# agent are averaged.
def rate(stats, agent, key="score"):
    rounds = len(stats["by_round"])
    copies = [v for name, v in stats["by_agent"].items()
              if name == agent or re.fullmatch(rf"{re.escape(agent)}_\d+", name)]
    if not copies:
        raise KeyError(agent)
    if key == "score":
        return mean((v["coins"] + 5 * v["kills"]) / v["rounds"] for v in copies)
    return mean(v.get(key, 0) / v["rounds"] for v in copies)


# Loads the stats files of the seat blocks of one evaluation.
def seats(paths):
    paths = sorted(paths)
    return [json.loads(Path(p).read_text()) for p in paths]


# Per-block values of two agents in the same games and their per-block differences.
def contrast(blocks, a, b, key="score"):
    x = [rate(s, a, key) for s in blocks]
    y = [rate(s, b, key) for s in blocks]
    d = [i - j for i, j in zip(x, y)]
    return x, y, d


# One markdown table row: both agents' means and the paired difference with its verdict.
def row(label, blocks, a, b):
    x, y, d = contrast(blocks, a, b)
    return f"| {label} | {fmt(x)} | {fmt(y)} | {mean(d):+.3f} +- {se(d):.3f} | {verdict(mean(d), se(d))} |"


# Section heading in REPORT_NUMBERS.md.
def header(title):
    print(f"\n## {title}\n")


# Marks a result whose files are missing.
def missing(what):
    print(f"_missing: {what}_\n")


# H1/H2: adapter determinism and zero-residual parity from the trace summary.
def parity():
    header("H1/H2 correctness gate: frozen adapter determinism and zero-residual parity")
    path = RUNS / "H1_H2_parity/summary.json"
    if not path.exists():
        return missing(path)
    s = json.loads(path.read_text())
    print(f"- {s['rounds']} fixed-seed classic rounds (seed 1201), {s['adapter_decisions']} adapter decisions")
    print(f"- adapter run A == run B: {s['adapter_run_a_equals_run_b']}")
    print(f"- zero-residual hybrid matches adapter: {s['hybrid_matches_adapter']}/{s['adapter_decisions']}")
    print(f"- parity exact: {s['parity_exact']}")


# H0: q-agent against its identical copy.
def noise_floor():
    header("H0 noise floor: q-agent against an identical copy of itself (4 seats x 200 rounds, seed 4801)")
    blocks = seats((RUNS / "H0_noise_floor_200").glob("null_seat*.json"))
    if len(blocks) != 4:
        return missing("H0_noise_floor_200")
    print("| comparison | q-agent v1 | q-agent v1 (identical copy) | difference | status |\n|---|---|---|---|---|")
    print(row("identical tables", blocks, Q_AGENT, "q_agent_v1_copy"))


# H1: adapter with full table, without table, and with WAIT for unseen states.
def adapter_ablation():
    header("H1 adapter ablation (25 rounds, seed 909, adapter + 3 rule_based)")
    folder = ROOT / "results/q_agent_hybrid/hybrid/runs/H1_adapter_ablation"
    print("| adapter mode | score/round | coins | kills | suicides |\n|---|---|---|---|---|")
    for mode in ("full", "no_q_table", "unseen_wait"):
        path = folder / f"{mode}_classic_25.json"
        if not path.exists():
            print(f"| {mode} | missing | | | |")
            continue
        s = json.loads(path.read_text())
        agent = next(a for a in s["by_agent"] if "adapter" in a)
        print(f"| {mode} | {rate(s, agent):.3f} | {rate(s, agent, 'coins'):.3f} | "
              f"{rate(s, agent, 'kills'):.3f} | {rate(s, agent, 'suicides'):.3f} |")


# H3-H8 screens against the q-agent, plus H4 pooled over its three training seeds.
def early_screens():
    header("H3-H8 early screens: hybrid vs q-agent, four 25-round seat blocks, SE over blocks")
    print("| stage | run | hybrid | q-agent | hybrid - q-agent | status |\n|---|---|---|---|---|---|")
    hybrid = "hybrid_h3_h8"
    screens = [
        ("H3", "first residual DDQN (seed 1401)", RUNS / "H3_residual_ddqn_seed1401", "eval_seat*.json"),
        ("H5-C", "conservative training, n=3", RUNS / "H5_conservative_seed1501", "eval_seat*.json"),
        ("H5-S", "potential shaping, n=3", RUNS / "H5_shaping_seed1501", "eval_seat*.json"),
        ("H6", "replay/training repair", RUNS / "H6_replay_repair_seed1501", "eval_seat*.json"),
        ("H7", "six-input residual", RUNS / "H7_six_inputs_seed1501", "eval_seat*.json"),
        ("H8-B", "antithetic ES, best vector", RUNS / "H8_antithetic_es_seed1901", "eval_best_seat*.json"),
        ("H8-M", "antithetic ES, mean vector", RUNS / "H8_antithetic_es_seed1901", "eval_mean_seat*.json"),
    ]
    h4 = [(f"H4-{n}", f"{n}-step, seed {seed}", RUNS / f"H4_nstep_equal_budget/n{n}_seed{seed}", "eval_seat*.json")
          for n in (1, 3) for seed in (1501, 1502, 1503)]
    screens = screens[:1] + h4 + screens[1:]
    for label, text, folder, pattern in screens:
        blocks = seats(folder.glob(pattern))
        if len(blocks) != 4:
            print(f"| {label} | {text} | missing | | | |")
            continue
        print(row(f"{label} | {text}", blocks, hybrid, Q_AGENT))
    header("H4 pooled over the three training seeds (12 seat blocks per return length)")
    print("| return | hybrid | q-agent | hybrid - q-agent | status |\n|---|---|---|---|---|")
    pooled = {}
    for n in (1, 3):
        blocks = []
        for seed in (1501, 1502, 1503):
            blocks += seats((RUNS / f"H4_nstep_equal_budget/n{n}_seed{seed}").glob("eval_seat*.json"))
        if len(blocks) != 12:
            print(f"| {n}-step | missing ({len(blocks)}/12 blocks) | | | |")
            continue
        pooled[n] = [mean(rate(s, hybrid) for s in blocks[i:i + 4]) for i in (0, 4, 8)]
        print(row(f"{n}-step", blocks, hybrid, Q_AGENT))
    if len(pooled) == 2:
        d = [b - a for a, b in zip(pooled[1], pooled[3])]
        print(f"\n3-step minus 1-step, paired by training seed (n=3 seeds): {mean(d):+.3f} +- {se(d):.3f} "
              f"({verdict(mean(d), se(d))}); per seed {', '.join(f'{x:+.3f}' for x in d)}")


# H8: how the hill-climbing and ES searches developed over their epochs.
def search_traces():
    header("H8 direct score search: optimisation traces")
    for label, folder, name in (("H8 hill climbing", "H8_hillclimb_seed1701", "acceptance.jsonl"),
                                ("H8 Adam-ES", "H8_adam_es_seed1801", "epochs.jsonl"),
                                ("H8 antithetic Adam-ES", "H8_antithetic_es_seed1901", "epochs.jsonl")):
        path = RUNS / folder / name
        if not path.exists():
            print(f"- {label}: missing")
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if name == "acceptance.jsonl":
            accepted = sum(bool(r.get("accepted")) for r in rows)
            print(f"- {label}: {len(rows)} iterations, {accepted} candidates accepted")
            continue
        keys = [k for k in rows[0] if "score" in k and isinstance(rows[0][k], (int, float))]
        key = "score" if "score" in keys else ("mean_score" if "mean_score" in keys else keys[0])
        first = [r[key] for r in rows[:100]]
        last = [r[key] for r in rows[-100:]]
        print(f"- {label}: {len(rows)} epochs; `{key}` first 100 epochs {fmt(first)}, last 100 {fmt(last)}")


# H9: score for every limit/margin combination.
def h9_sweep():
    header("H9 calibration sweep of the H7 checkpoint (200 rounds, seed 2201, roster H9, H7, q-agent, rule)")
    folder = RUNS / "H9_calibration_sweep_200"
    print("| limit L | margin m | budget 2L-m | H9 score | H7 (L=.25,m=.25) | q-agent |\n|---|---|---|---|---|---|")
    for limit in ("0.15", "0.25", "0.35"):
        for margin in ("0.10", "0.25", "0.40"):
            path = folder / f"limit{limit}_margin{margin}.json"
            if not path.exists():
                print(f"| {limit} | {margin} | | missing | | |")
                continue
            s = json.loads(path.read_text())
            print(f"| {limit} | {margin} | {2 * float(limit) - float(margin):.2f} | "
                  f"{rate(s, 'hybrid_h9'):.3f} | {rate(s, 'hybrid_h3_h8'):.3f} | {rate(s, Q_AGENT):.3f} |")


LADDER = [
    ("H9", "L=.25, m=.25, 6 inputs (H7 checkpoint)", "hybrid_h9", "H9_H12_ladder_200", "h9_seat*.json"),
    ("H10", "add scalar value head", "hybrid_h10", "H10_eval_200", "blockA_seat*.json"),
    ("H11", "add eight observations", "hybrid_h11", "H11_eval_200", "blockA_seat*.json"),
    ("H12", "L=1.00, m=.05, anchor .01", "hybrid_h12", "H9_H12_ladder_200", "h12_seat*.json"),
]


# H9-H12 in the common field: scores, differences to q-agent and rule-based agents, consecutive
# steps, behaviour.
def ladder():
    header("H9-H12 common ladder: {stage, rule, rule, q-agent}, classic, seed 4801, 4 seats x 200 rounds")
    print("| stage | change | stage score | rule_based | q-agent | stage - q-agent | status | stage - rule | status |")
    print("|---|---|---|---|---|---|---|---|---|")
    kept = []
    for label, text, agent, folder, pattern in LADDER:
        blocks = seats((RUNS / folder).glob(pattern))
        if len(blocks) != 4:
            print(f"| {label} | {text} | missing | | | | | | |")
            continue
        x, y, d = contrast(blocks, agent, Q_AGENT)
        _, r, dr = contrast(blocks, agent, RULE)
        kept.append((label, agent, blocks, x))
        print(f"| {label} | {text} | {fmt(x)} | {fmt(r)} | {fmt(y)} | {mean(d):+.3f} +- {se(d):.3f} | "
              f"{verdict(mean(d), se(d))} | {mean(dr):+.3f} +- {se(dr):.3f} | {verdict(mean(dr), se(dr))} |")
    print("\nConsecutive stages (unpaired, different games):\n")
    for (la, _, _, xa), (lb, _, _, xb) in zip(kept, kept[1:]):
        delta, error = mean(xb) - mean(xa), math.hypot(se(xa), se(xb))
        print(f"- {lb} - {la}: {delta:+.3f} +- {error:.3f} ({verdict(delta, error)})")
    print("\nBehaviour per round (coins, kills, suicides):\n")
    print("| stage | coins | kills | suicides |\n|---|---|---|---|")
    for label, agent, blocks, _ in kept:
        cells = [fmt([rate(s, agent, k) for s in blocks]) for k in ("coins", "kills", "suicides")]
        print(f"| {label} | " + " | ".join(cells) + " |")


# Direct games between consecutive stages.
def head_to_head():
    header("Direct head-to-head blocks (4 seats x 200 rounds, seed 4801)")
    print("| comparison | first | second | first - second | status |\n|---|---|---|---|---|")
    for text, folder, pattern, a, b in (
            ("H10 vs H9 (H10, H9, q-agent, rule)", "H10_eval_200", "blockB_seat*.json",
             "hybrid_h10", "hybrid_h9"),
            ("H11 vs H10 (H11, H10, q-agent, rule)", "H11_eval_200", "blockB_seat*.json",
             "hybrid_h11", "hybrid_h10"),
            ("H12 vs H9 (H12, H9, q-agent, rule)", "H12_eval_200", "target_seat*.json",
             "hybrid_h12", "hybrid_h9"),
            ("H12 vs q-agent (H12, H9, q-agent, rule)", "H12_eval_200", "target_seat*.json",
             "hybrid_h12", Q_AGENT),
            ("H12 vs rule (H12, H9, q-agent, rule)", "H12_eval_200", "target_seat*.json",
             "hybrid_h12", RULE)):
        blocks = seats((RUNS / folder).glob(pattern))
        if len(blocks) != 4:
            print(f"| {text} | missing | | | |")
            continue
        print(row(text, blocks, a, b))


# H12 vs q-agent over 6,400 rounds: seed-base estimator, t-interval and round-paired estimator.
def powered():
    header("Powered H12 vs q-agent: {H12, rule, rule, q-agent}, 8 seed bases x 4 seats x 200 rounds")
    folder = RUNS / "H12_vs_q_agent_powered"
    bases = defaultdict(list)
    for path in sorted(folder.glob("base*_seat[0-9].json")):
        bases[re.match(r"base(\d+)_seat", path.name).group(1)].append(path)
    complete = {b: p for b, p in bases.items() if len(p) == 4}
    if not complete:
        return missing(folder)
    h12 = "hybrid_h12"
    per_base = {b: (mean(rate(s, h12) for s in seats(p)), mean(rate(s, Q_AGENT) for s in seats(p)))
                for b, p in sorted(complete.items())}
    diffs = [h - d for h, d in per_base.values()]
    print(f"{len(complete)} complete seed bases.\n\n| base | H12 | q-agent | H12 - q-agent |\n|---|---|---|---|")
    for b, (h, d) in per_base.items():
        print(f"| {b} | {h:.3f} | {d:.3f} | {h - d:+.3f} |")
    hs, ds = [h for h, _ in per_base.values()], [d for _, d in per_base.values()]
    print(f"\n- seed-base estimator: H12 {fmt(hs)}, q-agent {fmt(ds)}, difference {mean(diffs):+.3f} +- "
          f"{se(diffs):.3f} ({verdict(mean(diffs), se(diffs))})")
    df = len(diffs) - 1
    if df in T975:
        half = T975[df] * se(diffs)
        print(f"- Student-t 95% interval over the {len(diffs)} base differences: "
              f"[{mean(diffs) - half:+.3f}, {mean(diffs) + half:+.3f}]")
    pairs = []
    for paths in complete.values():
        for path in paths:
            rounds_file = path.parent / (path.stem + ".rounds.json")
            if rounds_file.exists():
                for r in json.loads(rounds_file.read_text()):
                    hc, hk = r.get(h12, [0, 0])
                    dc, dk = r.get(Q_AGENT, [0, 0])
                    pairs.append(hc + 5 * hk - dc - 5 * dk)
    if pairs:
        sd = stdev(pairs)
        print(f"- round-paired estimator over {len(pairs)} rounds: {mean(pairs):+.3f} +- {se(pairs):.3f} "
              f"({verdict(mean(pairs), se(pairs))}); per-round sd {sd:.2f}")
        if abs(mean(pairs)) > 1e-9:
            print(f"- rounds needed to resolve an effect of this size: ~{math.ceil((2 * sd / abs(mean(pairs))) ** 2)}")
    blocks = [s for p in complete.values() for s in seats(p)]
    print("\n| agent | coins | kills | suicides |\n|---|---|---|---|")
    for agent in (h12, Q_AGENT):
        cells = [fmt([rate(s, agent, k) for s in blocks]) for k in ("coins", "kills", "suicides")]
        print(f"| {agent} | " + " | ".join(cells) + " |")


# H12 and the q-agent each against three rule-based agents on the same maps.
def generalization():
    header("Held-out roster: agent + 3 rule_based, seed 5801, 4 seats x 200 rounds (same maps for H12 and q-agent)")
    folder = RUNS / "H12_heldout_rules_200"
    h = seats(folder.glob("h12_seat*.json"))
    d = seats(folder.glob("q_agent_seat*.json"))
    if len(h) != 4 or len(d) != 4:
        return missing(folder)
    print("| agent | score | coins | kills | suicides |\n|---|---|---|---|---|")
    for name, blocks, agent in (("H12", h, "hybrid_h12"), ("q-agent", d, Q_AGENT)):
        cells = [fmt([rate(s, agent, k) for s in blocks]) for k in ("score", "coins", "kills", "suicides")]
        print(f"| {name} | " + " | ".join(cells) + " |")
    diff = [rate(a, "hybrid_h12") - rate(b, Q_AGENT) for a, b in zip(h, d)]
    print(f"\nseat-paired H12 - q-agent: {mean(diff):+.3f} +- {se(diff):.3f} ({verdict(mean(diff), se(diff))})")


# H12 retrained with seeds 1502/1503 in the training-like and the held-out field.
def replication():
    header("H12 training-seed replication (seeds 1502, 1503), eval seed 8201, 4 seats x 100 rounds")
    control = seats((RUNS / "H12_replication_q_agent_control").glob("heldout_rules_seat*.json"))
    print("| train seed | field | H12 | q-agent | H12 - q-agent | status |\n|---|---|---|---|---|---|")
    for seed in (1502, 1503):
        folder = RUNS / f"H12_replication_seed{seed}"
        mixed = seats(folder.glob("mixed_seat*.json"))
        if len(mixed) == 4:
            print(row(f"{seed} | H12, q-agent, rule, rule", mixed, "hybrid_h12", Q_AGENT))
        else:
            print(f"| {seed} | mixed | missing | | | |")
        held = seats(folder.glob("heldout_rules_seat*.json"))
        if len(held) == 4 and len(control) == 4:
            x = [rate(s, "hybrid_h12") for s in held]
            y = [rate(s, Q_AGENT) for s in control]
            d = [i - j for i, j in zip(x, y)]
            print(f"| {seed} | H12, rule, rule, rule | {fmt(x)} | {fmt(y)} | {mean(d):+.3f} +- {se(d):.3f} | "
                  f"{verdict(mean(d), se(d))} |")
        else:
            print(f"| {seed} | held-out | missing | | | |")
    print("\nBehaviour in the held-out field (coins, kills, suicides per round):\n")
    for seed in (1502, 1503):
        held = seats((RUNS / f"H12_replication_seed{seed}").glob("heldout_rules_seat*.json"))
        if len(held) == 4:
            cells = [f"{mean(rate(s, 'hybrid_h12', k) for s in held):.3f}"
                     for k in ("coins", "kills", "suicides")]
            print(f"- seed {seed}: " + ", ".join(cells))
    if len(control) == 4:
        cells = [f"{mean(rate(s, Q_AGENT, k) for s in control):.3f}" for k in ("coins", "kills", "suicides")]
        print("- q-agent control: " + ", ".join(cells))


# Last line of every training run's metrics (interactions, updates, epsilon, loss).
def training():
    header("Training diagnostics (final row of metrics.jsonl)")
    print("| run | rounds | interactions | updates | final epsilon | mean loss | mean round reward |")
    print("|---|---|---|---|---|---|---|")
    runs = ["H3_residual_ddqn_seed1401", "H5_conservative_seed1501", "H5_shaping_seed1501",
            "H6_replay_repair_seed1501", "H7_six_inputs_seed1501", "H10_dueling_seed1501",
            "H11_features_seed1501", "H12_authority_seed1501",
            "H12_replication_seed1502", "H12_replication_seed1503"]
    runs += [f"H4_nstep_equal_budget/n{n}_seed{s}" for n in (1, 3) for s in (1501, 1502, 1503)]
    for run in runs:
        path = RUNS / run / "metrics.jsonl"
        if not path.exists():
            print(f"| {run} | missing | | | | | |")
            continue
        last = json.loads(path.read_text().splitlines()[-1])
        get = lambda k, f="{}": f.format(last[k]) if last.get(k) is not None else "-"
        print(f"| {run} | {get('round')} | {get('interactions')} | {get('updates')} | "
              f"{get('epsilon', '{:.3f}')} | {get('mean_loss', '{:.3f}')} | {get('mean_round_reward', '{:.2f}')} |")


# How often each trained stage chooses a different action than the q-agent.
def override_rates():
    header("Residual override rate of the trained checkpoints (50 rounds, seed 4801)")
    print("hybrid_h3_h8 = H9 (H7 checkpoint, limit 0.25, margin 0.25)\n")
    print("| agent | decisions | override rate | mean largest correction |\n|---|---|---|---|")
    for path in sorted((RUNS / "override_rate").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        n = len(rows)
        print(f"| {path.stem} | {n} | {sum(r['residual_override'] for r in rows) / n:.4f} | "
              f"{sum(r['residual_max_abs'] for r in rows) / n:.3f} |")


if __name__ == "__main__":
    print("# Hybrid experiments: report numbers")
    print("Score per round = coins + 5 * kills. Mean +- standard error over seat blocks; resolved if |mean| > 2 SE.")
    for section in (parity, noise_floor, adapter_ablation, early_screens, search_traces, h9_sweep, ladder,
                    head_to_head, powered, generalization, replication, training, override_rates):
        try:
            section()
        except Exception as error:
            print(f"\n_error in {section.__name__}: {error!r}_\n", file=sys.stdout)
