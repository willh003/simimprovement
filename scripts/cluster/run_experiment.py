#!/gscratch/weirdlab/will/venvs/ogbench/bin/python
"""Launch a steering training run on weirdlab GPUs: an openpi server job + an Isaac apptainer training job.

  scripts/cluster/run_experiment.py NAME [--time HOURS] [--pi-addr HOST:PORT] [--dry-run] -- <train_steering.py args>
  e.g. scripts/cluster/run_experiment.py cb1 -- --environment CubeBowl --num-envs 128

Checks free weirdlab GPUs first (hyakalloc); submits nothing and exits 2 if they are not available. Comma-separated
partition lists don't work on Klone (sbatch silently uses the first), so partitions are tried one at a time.
Logs / state: runs/cluster/NAME/{pi.log,train.log,jobs.json,pi_server.addr}. See docs/claude/cluster_runs.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/bin"))
from show_weird_compute import quota_summary  # noqa: E402  rows: (partition, "free/total gpus", cpus, mem)

ROOT = Path(__file__).resolve().parents[2]
CL = ROOT / "scripts/cluster"
PI_PARTS = ["gpu-a40", "gpu-l40", "gpu-l40s"]  # pi server: any weirdlab gpu
TRAIN_PARTS = ["gpu-l40", "gpu-l40s"]  # Isaac: L40 preferred (same arch 8.9 on l40s)


def free_gpus() -> dict[str, int]:
    return {r[0]: int(r[1].split("/")[0]) for r in quota_summary()}


def sbatch(script, part, name, exp_dir, log, hours, env, extra=()):
    cmd = ["sbatch", "--parsable", f"--partition={part}", f"--job-name={name}", f"--time={hours}:00:00", f"--output={exp_dir/log}",
           "--export=ALL", *extra, str(CL / script)]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
    if r.returncode:
        print(f"sbatch failed on {part}: {r.stderr.strip()}")
        return None
    return r.stdout.strip().split(";")[0]


def main():
    argv = sys.argv[1:]
    train_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, train_args = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--time", type=int, default=12, help="hours for the training job (pi server gets +1)")
    p.add_argument("--pi-addr", help="reuse a running server instead of launching one")
    p.add_argument("--dry-run", action="store_true", help="only report availability")
    a = p.parse_args(argv)

    exp_dir = ROOT / "runs/cluster" / a.name
    if (exp_dir / "jobs.json").exists():
        ids = list(json.loads((exp_dir / "jobs.json").read_text()).values())
        live = subprocess.run(["squeue", "-h", "-j", ",".join(map(str, ids)), "-o", "%i"], capture_output=True, text=True).stdout.split()
        if live:
            sys.exit(f"experiment {a.name} still has live jobs {live}; stop it with exp_stop.sh or pick another name")

    need_pi = not a.pi_addr and "--dummy-policy" not in train_args
    free = free_gpus()
    print("free weirdlab GPUs:", free)
    train_part = next((x for x in TRAIN_PARTS if free.get(x, 0) > 0), None)
    if train_part:
        free[train_part] -= 1
    pi_part = next((x for x in PI_PARTS if free.get(x, 0) > 0), None) if need_pi else None
    if train_part is None or (need_pi and pi_part is None):
        sys.exit(f"NOT AVAILABLE: need {'an L40/L40s' if train_part is None else ''}"
                 f"{' and ' if train_part is None and need_pi and pi_part is None else ''}"
                 f"{'a weirdlab GPU for the pi server' if need_pi and pi_part is None else ''} free; nothing submitted (run `swc`).")
    print(f"plan: train on {train_part}" + (f", pi server on {pi_part}" if need_pi else f", pi server: {a.pi_addr or 'none (dummy policy)'}"))
    if a.dry_run:
        return 0

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "pi_server.addr").unlink(missing_ok=True)
    if a.pi_addr:
        (exp_dir / "pi_server.addr").write_text(a.pi_addr + "\n")
    env = {"EXP_DIR": str(exp_dir)}
    jobs = {}
    if need_pi:
        pi_id = None
        for part in [x for x in PI_PARTS if free.get(x, 0) > 0]:  # fall back to the next partition if rejected
            pi_id = sbatch("pi_server_job.sbatch", part, f"pi-{a.name}", exp_dir, "pi.log", a.time + 1, env)
            if pi_id:
                break
        if not pi_id:
            sys.exit("could not submit the pi server job")
        jobs["pi"] = pi_id
    train_env = {**env, "TRAIN_ARGS": shlex.join(train_args), "PI_MODE": "wait" if (need_pi or a.pi_addr) else "none"}
    if need_pi:
        train_env["PI_JOB_ID"] = jobs["pi"]
    train_id = None
    for part in [x for x in TRAIN_PARTS if x == train_part or free.get(x, 0) > 0]:
        train_id = sbatch("train_job.sbatch", part, f"train-{a.name}", exp_dir, "train.log", a.time, train_env)
        if train_id:
            break
    if not train_id:
        if "pi" in jobs:
            subprocess.run(["scancel", jobs["pi"]])
        sys.exit("could not submit the training job")
    jobs["train"] = train_id
    (exp_dir / "jobs.json").write_text(json.dumps(jobs))
    print(f"submitted {jobs}; logs in {exp_dir}")

    time.sleep(20)
    for role, jid in jobs.items():
        st = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%T %N %r"], capture_output=True, text=True).stdout.strip()
        print(f"  {role} job {jid}: {st or 'finished/not in queue'}")
    print("check progress with scripts/cluster/exp_status.sh", a.name)


if __name__ == "__main__":
    sys.exit(main())
