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


def compare(args):
    from .site import PRICE_IN, PRICE_OUT
    s = json.load(open(SPLIT, encoding="utf-8"))
    rows = trials(args.job)
    new = per_task(rows)
    tasks = sorted(new)
    base = s["baseline"]
    outside = [t for t in tasks if t not in base]
    if outside:
        sys.exit("tasks with no baseline: %s" % outside)
    halves = sorted(set(h for t in tasks for h in ("dev", "heldout") if t in s[h]))
    rate = lambda b: b["passes"] / b["attempts"]
    diffs = [rate(new[t]) - rate(base[t]) for t in tasks]
    rng, boots = random.Random(0), []
    for _ in range(4000):
        pick = [rng.choice(diffs) for _ in diffs]
        boots.append(sum(pick) / len(pick))
    boots.sort()
    n = len(tasks)
    if args.table:
        print("%-34s %9s %9s  %-6s %5s %9s  %s" % ("task", "stock", "layer", "turns", "recov", "timeouts", "claim decisions"))
        for t in tasks:
            rs = [r for r in rows if r["task"] == t]
            dec = [" > ".join(rd["decision"] for rd in r["checks"]["rounds"]) or "-" for r in rs if r["checks"]]
            print("%-34s %4d/%-4d %4d/%-4d  %-6s %5s %9d  %s" % (
                t[:34], base[t]["passes"], base[t]["attempts"], new[t]["passes"], new[t]["attempts"],
                "/".join(str(r["turns"]) for r in rs), "/".join(str((r["checks"] or {}).get("recovered_turns", "-")) for r in rs),
                sum(r["timeout"] for r in rs), " | ".join(dec)))
    print("## %s: %d tasks from %s, %d trials with a verdict" % (Path(args.job).name, n, "+".join(halves), len(rows)))
    print("   pass rate (mean over tasks): stock %.3f -> layer %.3f   paired difference %+.3f  95%% CI [%+.3f, %+.3f]"
          % (sum(rate(base[t]) for t in tasks) / n, sum(rate(new[t]) for t in tasks) / n,
             sum(diffs) / n, boots[100], boots[3899]))
    gained = [t for t in tasks if new[t]["passes"] and not base[t]["passes"]]
    lost = [t for t in tasks if base[t]["passes"] and not new[t]["passes"]]
    print("   solved only with the layer: %s" % (", ".join(gained) or "none"))
    print("   solved only by stock:       %s" % (", ".join(lost) or "none"))
    bc, bf = sum(base[t]["claims"] for t in tasks), sum(base[t]["false_claims"] for t in tasks)
    nc, nf = sum(new[t]["claims"] for t in tasks), sum(new[t]["false_claims"] for t in tasks)
    print("   runs ending in a claim that failed the tests: stock %d/%d (%.0f%%) -> layer %d/%d (%.0f%%)"
          % (bf, bc, 100.0 * bf / max(bc, 1), nf, nc, 100.0 * nf / max(nc, 1)))

    logged = [r for r in rows if r["checks"] is not None]
    if logged:
        decisions = {}
        for r in logged:
            for rd in r["checks"]["rounds"]:
                decisions[rd["decision"]] = decisions.get(rd["decision"], 0) + 1
        written = [len([c for c in (r["checks"]["rounds"][0]["results"] if r["checks"]["rounds"] else [])]) for r in logged]
        no_checks = sum(1 for r in logged if r["checks"]["rounds"] and not any(
            c["status"] in ("pass", "fail") for c in r["checks"]["rounds"][0]["results"]))
        print("   layer: %d trials, claim decisions %s, usable checks missing in %d, checks per trial %.1f"
              % (len(logged), decisions, no_checks, sum(written) / len(logged)))
        # Does the layer's last verdict agree with the hidden tests?
        m = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
        for r in logged:
            usable = [c for c in (r["checks"]["rounds"][-1]["results"] if r["checks"]["rounds"] else [])
                      if c["status"] in ("pass", "fail")]
            if usable:
                m[(all(c["status"] == "pass" for c in usable), r["passed"])] += 1
        print("   last verdict vs hidden tests: checks pass & tests pass %d, checks pass & tests fail %d, "
              "checks fail & tests pass %d, checks fail & tests fail %d"
              % (m[(True, True)], m[(True, False)], m[(False, True)], m[(False, False)]))

    turns = lambda rs: sum(r["turns"] for r in rs) / max(len(rs), 1)
    usd = lambda rs: sum(r["tokens_in"] * PRICE_IN + r["tokens_out"] * PRICE_OUT for r in rs) / max(len(rs), 1)
    bj = ROOT / "jobs" / s["baseline_job"]
    if bj.exists():
        brows = [r for r in trials(bj) if r["task"] in new]
        print("   per trial: stock %.1f turns, $%.4f, %d timeouts in %d  ->  layer %.1f turns, $%.4f, %d timeouts in %d"
              % (turns(brows), usd(brows), sum(r["timeout"] for r in brows), len(brows),
                 turns(rows), usd(rows), sum(r["timeout"] for r in rows), len(rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    c = sub.add_parser("compare")
    c.add_argument("job", help="job dir of the layered agent")
    c.add_argument("--table", action="store_true", help="one row per task first")
    args = ap.parse_args()
    {"split": split, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
