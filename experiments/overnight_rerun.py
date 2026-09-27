# Runs the whole experiment list: a few jobs at a time (--jobs, default 2), each job starts once
# its checkpoints and prerequisite jobs are there.
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/q_agent_hybrid/hybrid/runs"
CK = ROOT / "results/q_agent_hybrid/hybrid/checkpoints"
STATUS = RUNS / "rerun_logs"


# Path of a checkpoint file.
def ck(name: str) -> Path:
    return CK / f"{name}.pt"


JOBS: dict[str, tuple[str, list[Path], list[str], list[str]]] = {}


# Registers one job: shell command, files it needs (needs=), jobs it waits for (after=), folders
# to clear before it starts (wipe=).
def job(name, command, needs=(), after=(), wipe=()):
    JOBS[name] = (command, list(needs), list(after), list(wipe))


job("h3_train", "bash experiments/run_q_agent_h3.sh")
job("h5_conservative", "bash experiments/run_q_agent_h5_conservative.sh")
job("h5_shaping", "bash experiments/run_q_agent_h5_shaping.sh")
job("h6_train", "bash experiments/run_q_agent_h6_replay_repair.sh")
job("h7_train", "bash experiments/run_q_agent_h7_six_inputs.sh")
job("h11_train", "bash experiments/run_q_agent_h11_features.sh", wipe=["H11_features_seed1501"])
job("h12_train", "bash experiments/run_q_agent_h12_authority.sh", wipe=["H12_authority_seed1501"])
job("h8_antithetic_es", "ANTITHETIC_ES_EPOCHS=1000 ANTITHETIC_ES_ROUNDS=5 .MLE/bin/python experiments/run_q_agent_h8_antithetic_es.py",
    wipe=["H8_antithetic_es_seed1901"])
job("h12_seed1502", "SEED=1502 TAG=H12_replication_seed1502 bash experiments/run_q_agent_h12_authority.sh",
    wipe=["H12_replication_seed1502"])
job("h12_seed1503", "SEED=1503 TAG=H12_replication_seed1503 bash experiments/run_q_agent_h12_authority.sh",
    wipe=["H12_replication_seed1503"])
job("h10_train", "bash experiments/run_q_agent_h10_dueling.sh", needs=[ck("H7_six_inputs_seed1501")], after=["h7_train"],
    wipe=["H10_dueling_seed1501"])
for n in (1, 3):
    for seed in (1501, 1502, 1503):
        job(f"h4_n{n}_s{seed}", f"N_STEP={n} SEED={seed} bash experiments/run_q_agent_h4_arm.sh")
job("h8_adam_es", "ADAM_ES_EPOCHS=1000 ADAM_ES_ROUNDS=5 .MLE/bin/python experiments/run_q_agent_h8_adam_es.py",
    wipe=["H8_adam_es_seed1801"])
job("h8_hillclimb", ".MLE/bin/python experiments/run_q_agent_h8_hillclimb.py", wipe=["H8_hillclimb_seed1701"])

job("h1_h2_parity", "bash experiments/run_q_agent_h1_h2_parity.sh")
job("h0_noise", "bash experiments/evaluate_h0_noise_floor.sh", wipe=["H0_noise_floor_200"])
job("h1_adapter", "bash experiments/evaluate_h1_adapter_ablation.sh")

H7, H10, H11, H12 = (ck("H7_six_inputs_seed1501"), ck("H10_dueling_seed1501"),
                     ck("H11_features_seed1501"), ck("H12_authority_seed1501"))
job("h8_antithetic_eval", "bash experiments/evaluate_h8_antithetic_es.sh", after=["h8_antithetic_es"])
job("h9_sweep", "bash experiments/run_q_agent_h9_calibration_sweep.sh", needs=[H7], after=["h7_train"],
    wipe=["H9_calibration_sweep_200"])
job("h10_eval", "bash experiments/evaluate_h10_dueling.sh", needs=[H7, H10], after=["h7_train", "h10_train"],
    wipe=["H10_eval_200"])
job("h11_eval", "bash experiments/evaluate_h11_features.sh", needs=[H10, H11], after=["h10_train", "h11_train"],
    wipe=["H11_eval_200"])
job("h12_target_eval", "bash experiments/evaluate_h12_target.sh", needs=[H7, H12], after=["h7_train", "h12_train"],
    wipe=["H12_eval_200"])
job("ladder_h9", "ARMS=h9 bash experiments/evaluate_h9_h12_ladder.sh", needs=[H7], after=["h7_train"])
job("ladder_h12", "ARMS=h12 bash experiments/evaluate_h9_h12_ladder.sh", needs=[H12], after=["h12_train"])
for part, bases in enumerate(("6001 6101", "6201 6301", "6401 6501", "6601 6701")):
    job(f"powered_{part}", f"BASES='{bases}' bash experiments/evaluate_h12_vs_q_agent_powered.sh", needs=[H12], after=["h12_train"])
job("generalization", "bash experiments/evaluate_h12_heldout_rules.sh", needs=[H12], after=["h12_train"],
    wipe=["H12_heldout_rules_200", "H12_replication_q_agent_control"])
job("rep1502_eval", "TRAIN_SEED=1502 bash experiments/evaluate_h12_replication.sh", after=["h12_seed1502"])
job("rep1503_eval", "TRAIN_SEED=1503 bash experiments/evaluate_h12_replication.sh", after=["h12_seed1503"])

job("override_rates", "bash experiments/measure_override_rate.sh", needs=[H7, H10, H11, H12],
    after=["h7_train", "h10_train", "h11_train", "h12_train"])

WIPE_AT_START = ["H9_H12_ladder_200", "H12_vs_q_agent_powered", "H4_nstep_equal_budget"]


# Timestamped line in scheduler.log.
def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with (STATUS / "scheduler.log").open("a") as handle:
        handle.write(line + "\n")


# Scheduler loop: every 20 s it collects finished jobs and, while fewer than --jobs are running,
# starts every job whose dependencies are done. Ends when all jobs are finished and writes
# ALL_DONE.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=2, help="how many jobs run at the same time")
    parser.add_argument("--only", nargs="*", help="run only these jobs (debugging)")
    args = parser.parse_args()
    STATUS.mkdir(parents=True, exist_ok=True)
    for name in WIPE_AT_START if not args.only else ():
        subprocess.run(["rm", "-rf", str(RUNS / name)], check=True)
    pending = [name for name in JOBS if not args.only or name in args.only]
    running: dict[str, subprocess.Popen] = {}
    done: set[str] = set()
    failed: set[str] = set()
    log(f"scheduler start, {len(pending)} jobs, {args.jobs} at a time")
    while pending or running:
        for name, proc in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[name]
            (done if code == 0 else failed).add(name)
            (STATUS / f"{name}.{'ok' if code == 0 else 'failed'}").touch()
            log(f"{'DONE  ' if code == 0 else 'FAILED'} {name} (exit {code})")
        for name in list(pending):
            if len(running) >= args.jobs:
                break
            command, needs, after, wipe = JOBS[name]
            if any(dep in failed for dep in after):
                pending.remove(name); failed.add(name)
                (STATUS / f"{name}.failed").touch()
                log(f"SKIP   {name}: a prerequisite failed")
                continue
            if any(dep in JOBS and dep not in done and (not args.only or dep in args.only) for dep in after):
                continue
            if any(not path.exists() for path in needs):
                continue
            pending.remove(name)
            for directory in wipe:
                subprocess.run(["rm", "-rf", str(RUNS / directory)], check=True)
            env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1", "Q_AGENT_H3_CPU_THREADS": "1",
                                "PYTHONPATH": str(ROOT)}
            out = (STATUS / f"{name}.out").open("w")
            proc = subprocess.Popen(["bash", "-c", command], cwd=ROOT, env=env, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            running[name] = proc
            log(f"START  {name}: {command}")
        time.sleep(20)
    log(f"ALL JOBS FINISHED: {len(done)} ok, {len(failed)} failed {sorted(failed)}")
    (STATUS / ("ALL_DONE" if not failed else "ALL_DONE_WITH_FAILURES")).write_text("\n".join(sorted(failed)) + "\n")


if __name__ == "__main__":
    main()
