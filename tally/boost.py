"""Does a layer around the agent make a small model pass more Terminal-Bench tasks?

  python -m tally.boost split             lock the dev / held-out tasks (once)
  python -m tally.boost compare JOB_DIR   paired comparison with the stock-agent baseline

The split is drawn once from the tasks the stock agent ran, stratified by whether it
ever solved the task, and written to splits/terminalbench.json with the baseline's
per-task record. The layer is tuned on dev only. Held-out is run once, at the end,
and that is the number reported.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = "tally-terminalbench-run-nvidia-NVIDIA-Nemotron-3-Nano-30B-A3B-20260912-165615"
SPLIT = ROOT / "splits" / "terminalbench.json"
SEED = 20260913

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def trials(job_dir):
    """One record per trial with a verdict: task, passed, whether the agent's last move was
    a completion claim, turns, tokens, and the layer's log when the agent kept one."""
    out = []
    for d in sorted(p for p in Path(job_dir).iterdir() if p.is_dir() and "__" in p.name):
        f = d / "result.json"
        if not f.exists() or f.stat().st_size == 0:
            continue
        r = json.load(f.open(encoding="utf-8"))
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if reward is None:
            continue
        steps, tj = [], d / "agent" / "trajectory.json"
        if tj.exists():
            steps = json.load(tj.open(encoding="utf-8", errors="replace")).get("steps", [])
        agent_steps = [s for s in steps if s.get("source") == "agent"]
        last_calls = [c.get("function_name") for c in (agent_steps[-1].get("tool_calls") or [])] if agent_steps else []
        ar = r.get("agent_result") or {}
        out.append({
            "task": d.name.split("__")[0], "trial": d.name,
            "passed": reward >= 1.0,
            "claimed": "mark_task_complete" in last_calls,
            "timeout": ((r.get("exception_info") or {}).get("exception_type")) == "AgentTimeoutError",
            "turns": len(agent_steps),
            "tokens_in": ar.get("n_input_tokens") or 0, "tokens_out": ar.get("n_output_tokens") or 0,
            "checks": (ar.get("metadata") or {}).get("verification"),
        })
    return out


def per_task(rows):
    by = {}
    for r in rows:
        b = by.setdefault(r["task"], {"attempts": 0, "passes": 0, "claims": 0, "false_claims": 0})
        b["attempts"] += 1
        b["passes"] += r["passed"]
        b["claims"] += r["claimed"]
        b["false_claims"] += r["claimed"] and not r["passed"]
    return by


def split(args):
    if SPLIT.exists():
        sys.exit("%s already exists; the split is drawn once so held-out stays unseen" % SPLIT)
    job = ROOT / "jobs" / BASELINE
    base = per_task(trials(job))
    solved = sorted(t for t, b in base.items() if b["passes"])
    unsolved = sorted(t for t, b in base.items() if not b["passes"])
    rng, dev, held = random.Random(SEED), [], []
    for group in (solved, unsolved):
        g = list(group)
        rng.shuffle(g)
        half = (len(g) + 1) // 2
        dev += g[:half]
        held += g[half:]
    SPLIT.parent.mkdir(exist_ok=True)
    data = {
        "benchmark": "terminal-bench@2.0", "seed": SEED, "baseline_job": BASELINE,
        "rule": "tasks the stock agent ran, stratified by ever-solved, seeded shuffle, first half of each stratum to dev",
        "dev": sorted(dev), "heldout": sorted(held),
        "baseline": dict((t, base[t]) for t in sorted(base)),
    }
    SPLIT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    for name, ts in (("dev", dev), ("heldout", held)):
        a = sum(base[t]["attempts"] for t in ts)
        p = sum(base[t]["passes"] for t in ts)
        c = sum(base[t]["claims"] for t in ts)
        fc = sum(base[t]["false_claims"] for t in ts)
        print("%-8s %2d tasks (%d ever solved), baseline %d/%d attempts passed (%.3f), %d/%d claims false (%.0f%%)"
              % (name, len(ts), sum(1 for t in ts if base[t]["passes"]), p, a, p / a, fc, c, 100.0 * fc / max(c, 1)))
    print("wrote", SPLIT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    args = ap.parse_args()
    {"split": split}[args.cmd](args)


if __name__ == "__main__":
    main()
