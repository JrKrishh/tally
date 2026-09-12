"""tally run: execute a plan for real -- Harbor drives the agent on the planned cells.

The model is served by Nebius Token Factory (OpenAI-compatible), the agent is
Harbor's Terminus-2, each task runs in its own Docker sandbox, and only the
cells the plan chose are executed. Two phases, two Harbor jobs:

  run       the uncertain tasks, at the plan's attempt cap
  validate  a few tasks history called certain, once each, to check the imputation

  python -m tally.run --plan data/plan_terminalbench.json --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B --dry-run
  python -m tally.run --plan data/plan_terminalbench.json --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B --limit 1 --attempts 1
  python -m tally.run --plan data/plan_terminalbench.json --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B --phase all

The key is read from NEBIUS_API_KEY (or .nebius_key) and handed to Harbor's
process environment as OPENAI_API_KEY. It never appears on a command line or
in a config file. --dry-run needs no key and no Docker.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import nebius

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent


def harbor_bin():
    for c in (ROOT / ".venv-harbor" / "Scripts" / "harbor.exe", ROOT / ".venv-harbor" / "bin" / "harbor",
              Path.home() / ".local" / "bin" / "harbor"):        # uv tool install; not on PATH in non-login ssh
        if c.exists():
            return str(c)
    found = shutil.which("harbor")
    if not found:
        sys.exit("harbor not found: pip install harbor (Python 3.13) or uv tool install harbor")
    return found


MIN_FREE_GB = 30


def preflight(args):
    """Fail in seconds, not after 237 errored trials: the key must answer, the model
    must exist, and the disk must have room for task images."""
    free_gb = shutil.disk_usage(args.jobs_dir if os.path.isdir(args.jobs_dir) else str(ROOT)).free / 1e9
    if free_gb < MIN_FREE_GB:
        sys.exit("preflight: only %.0f GB free where jobs are written; need %d GB for task images" % (free_gb, MIN_FREE_GB))
    try:
        text, _, usage = nebius.chat(args.model, "Reply with the single word OK.", max_tokens=8)
    except SystemExit as e:
        sys.exit("preflight: the model call failed before any trial started -> %s" % e)
    print("## preflight ok: %s answered (%s tokens), %.0f GB free" % (args.model, usage.get("total_tokens", "?"), free_gb))


def build(plan, phase, args):
    tasks = list(plan["tasks_run"]) + list(plan.get("tasks_no_history", [])) if phase == "run" else list(plan["tasks_validate"])
    if args.limit:
        tasks = tasks[:args.limit]
    attempts = (args.attempts or plan["attempts"]) if phase == "run" else 1
    slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-")
    # Harbor writes a job dir even for a dry run and locks it; a fixed name would
    # collide on the next run. Timestamp every job; report reads the newest.
    job = "tally-%s-%s-%s-%s%s" % (plan["benchmark"], phase, slug,
                                   time.strftime("%Y%m%d-%H%M%S"), "-dry" if args.dry_run else "")
    if args.job_name:
        job = args.job_name              # reuse an existing job dir: Harbor resumes its unfinished trials
    cmd = [harbor_bin(), "run",
           "-d", plan["dataset"],
           "-a", "terminus-2",
           "-m", "openai/" + args.model,
           "--ak", "api_base=" + nebius.API,
           "--ak", "max_turns=%d" % args.max_turns,
           "--ak", "store_all_messages=true",
           "-k", str(attempts),
           "-n", str(args.concurrent),
           "-o", args.jobs_dir,
           "--job-name", job,
           "--max-retries", "1",                     # one retry on transient infra errors
           "--retry-exclude", "AgentTimeoutError",   # a timeout is a real outcome, not a hiccup
           "-y"]
    if args.max_thinking:
        cmd += ["--ak", "max_thinking_tokens=%d" % args.max_thinking]
    for t in tasks:
        cmd += ["-i", t]
    if args.dry_run:
        cmd.append("--dry-run")
    return job, tasks, attempts, cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", help="Token Factory model id")
    ap.add_argument("--phase", choices=["run", "validate", "all"], default="run")
    ap.add_argument("--attempts", type=int, help="override the plan's attempt cap for the run phase")
    ap.add_argument("--limit", type=int, help="first N tasks only (smoke test)")
    ap.add_argument("-n", "--concurrent", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-thinking", type=int, default=2048)
    ap.add_argument("--jobs-dir", default=str(ROOT / "jobs"))
    ap.add_argument("--job-name", help="resume an existing job dir: Harbor runs only trials with no result yet. "
                                       "Trials that already ERRORED count as done and are not retried; "
                                       "delete their dirs first, or start a fresh job")
    ap.add_argument("--dry-run", action="store_true", help="validate config and task names; no Docker, no key")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    if not args.dry_run:
        preflight(args)
    env = dict(os.environ)
    # Harbor writes trial results with Path.write_text() and no encoding. On Windows
    # that is cp1252, and the first model reply containing a character outside it
    # (a "≈" was enough) crashed a finished 11-minute trial at the final write.
    # Python's UTF-8 mode makes every default-encoded open() UTF-8.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if not args.dry_run:
        env["OPENAI_API_KEY"] = nebius.key()
    else:
        env.setdefault("OPENAI_API_KEY", "dry-run")

    phases = ["run", "validate"] if args.phase == "all" else [args.phase]
    for phase in phases:
        job, tasks, attempts, cmd = build(plan, phase, args)
        print("## %s: %d tasks x %d attempts -> %s%s" % (phase, len(tasks), attempts, job, "  [dry run]" if args.dry_run else ""))
        print("   " + " ".join(c if " " not in c else '"%s"' % c for c in cmd[1:]))
        rc = subprocess.call(cmd, env=env, cwd=str(ROOT))
        if rc != 0:
            sys.exit("harbor exited with %d during phase %s" % (rc, phase))
        print("## %s done -> %s" % (phase, Path(args.jobs_dir) / job))


if __name__ == "__main__":
    main()
