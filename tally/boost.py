"""Does a layer around the agent make a small model pass more Terminal-Bench tasks?

  python -m tally.boost split                     lock the dev / held-out tasks (once)
  python -m tally.boost compare JOB_DIR           paired comparison with the stock-agent baseline
  python -m tally.boost checkeval JOB_DIR         grade check writers against reference solutions
  python -m tally.boost thresholds EVAL_JOB DEV_JOB   reject on any failing check, or on a share of them?
  python -m tally.boost misfiled [JOB_DIR]        turns that came back with the answer filed as reasoning

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
NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
PRICES = {  # $/M prompt, completion -- GET /v1/models?verbose=true, 2026-09-13
    NANO: (0.06, 0.24),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
}

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


def paired(job_dir):
    """The layered job against the stock baseline on the same tasks: pass rate as the mean over tasks of
    each task's pass fraction, a paired difference with a 95% bootstrap interval over tasks, and what the
    layer did."""
    s = json.load(open(SPLIT, encoding="utf-8"))
    rows = trials(job_dir)
    new = per_task(rows)
    tasks = sorted(new)
    base = s["baseline"]
    outside = [t for t in tasks if t not in base]
    if outside:
        sys.exit("tasks with no baseline: %s" % outside)
    rate = lambda b: b["passes"] / b["attempts"]
    diffs = [rate(new[t]) - rate(base[t]) for t in tasks]
    rng, boots = random.Random(0), []
    for _ in range(4000):
        pick = [rng.choice(diffs) for _ in diffs]
        boots.append(sum(pick) / len(pick))
    boots.sort()
    n = len(tasks)
    out = {
        "job": Path(job_dir).name, "halves": sorted(set(h for t in tasks for h in ("dev", "heldout") if t in s[h])),
        "tasks": n, "trials": len(rows),
        "stock": sum(rate(base[t]) for t in tasks) / n, "layer": sum(rate(new[t]) for t in tasks) / n,
        "diff": sum(diffs) / n, "ci": [boots[100], boots[3899]],
        "stock_passes": [sum(base[t]["passes"] for t in tasks), sum(base[t]["attempts"] for t in tasks)],
        "layer_passes": [sum(new[t]["passes"] for t in tasks), sum(new[t]["attempts"] for t in tasks)],
        "gained": [t for t in tasks if new[t]["passes"] and not base[t]["passes"]],
        "lost": [t for t in tasks if base[t]["passes"] and not new[t]["passes"]],
        "false_claims": {"stock": [sum(base[t]["false_claims"] for t in tasks), sum(base[t]["claims"] for t in tasks)],
                         "layer": [sum(new[t]["false_claims"] for t in tasks), sum(new[t]["claims"] for t in tasks)]},
        "rows": [dict(task=t, stock=[base[t]["passes"], base[t]["attempts"]], layer=[new[t]["passes"], new[t]["attempts"]],
                      trials=[r for r in rows if r["task"] == t]) for t in tasks],
    }
    logged = [r for r in rows if r["checks"] is not None]
    if logged:
        decisions = {}
        for r in logged:
            for rd in r["checks"]["rounds"]:
                decisions[rd["decision"]] = decisions.get(rd["decision"], 0) + 1
        # Does the layer's last verdict agree with the hidden tests?
        m = {"pass_pass": 0, "pass_fail": 0, "fail_pass": 0, "fail_fail": 0}
        for r in logged:
            usable = [c for c in (r["checks"]["rounds"][-1]["results"] if r["checks"]["rounds"] else [])
                      if c["status"] in ("pass", "fail")]
            if usable:
                m["%s_%s" % ("pass" if all(c["status"] == "pass" for c in usable) else "fail",
                             "pass" if r["passed"] else "fail")] += 1
        out.update(decisions=decisions, verdicts=m,
                   recovered_turns=sum(r["checks"].get("recovered_turns", 0) for r in logged),
                   checks_per_trial=sum(len(r["checks"]["rounds"][0]["results"]) if r["checks"]["rounds"] else 0
                                        for r in logged) / len(logged),
                   no_usable_checks=sum(1 for r in logged if r["checks"]["rounds"] and not any(
                       c["status"] in ("pass", "fail") for c in r["checks"]["rounds"][0]["results"])))
    pi, po = PRICES[NANO]
    summary = lambda rs: {"trials": len(rs), "turns": sum(r["turns"] for r in rs) / max(len(rs), 1),
                          "usd": sum(r["tokens_in"] * pi + r["tokens_out"] * po for r in rs) / 1e6 / max(len(rs), 1),
                          "timeouts": sum(r["timeout"] for r in rs),
                          "tokens": [sum(r["tokens_in"] for r in rs), sum(r["tokens_out"] for r in rs)]}
    out["per_trial"] = {"layer": summary(rows)}
    bj = ROOT / "jobs" / s["baseline_job"]
    if bj.exists():
        out["per_trial"]["stock"] = summary([r for r in trials(bj) if r["task"] in new])
    return out


def compare(args):
    p = paired(args.job)
    if args.table:
        print("%-34s %9s %9s  %-6s %5s %9s  %s" % ("task", "stock", "layer", "turns", "recov", "timeouts", "claim decisions"))
        for row in p["rows"]:
            rs = row["trials"]
            dec = [" > ".join(rd["decision"] for rd in r["checks"]["rounds"]) or "-" for r in rs if r["checks"]]
            print("%-34s %4d/%-4d %4d/%-4d  %-6s %5s %9d  %s" % (
                row["task"][:34], row["stock"][0], row["stock"][1], row["layer"][0], row["layer"][1],
                "/".join(str(r["turns"]) for r in rs), "/".join(str((r["checks"] or {}).get("recovered_turns", "-")) for r in rs),
                sum(r["timeout"] for r in rs), " | ".join(dec)))
    print("## %s: %d tasks from %s, %d trials with a verdict" % (p["job"], p["tasks"], "+".join(p["halves"]), p["trials"]))
    print("   pass rate (mean over tasks): stock %.3f -> layer %.3f   paired difference %+.3f  95%% CI [%+.3f, %+.3f]"
          % (p["stock"], p["layer"], p["diff"], p["ci"][0], p["ci"][1]))
    print("   solved only with the layer: %s" % (", ".join(p["gained"]) or "none"))
    print("   solved only by stock:       %s" % (", ".join(p["lost"]) or "none"))
    fs, fl = p["false_claims"]["stock"], p["false_claims"]["layer"]
    print("   runs ending in a claim that failed the tests: stock %d/%d (%.0f%%) -> layer %d/%d (%.0f%%)"
          % (fs[0], fs[1], 100.0 * fs[0] / max(fs[1], 1), fl[0], fl[1], 100.0 * fl[0] / max(fl[1], 1)))
    if "decisions" in p:
        v = p["verdicts"]
        print("   layer: %d trials, claim decisions %s, usable checks missing in %d, checks per trial %.1f"
              % (p["trials"], p["decisions"], p["no_usable_checks"], p["checks_per_trial"]))
        print("   last verdict vs hidden tests: checks pass & tests pass %d, checks pass & tests fail %d, "
              "checks fail & tests pass %d, checks fail & tests fail %d"
              % (v["pass_pass"], v["pass_fail"], v["fail_pass"], v["fail_fail"]))
    if "stock" in p["per_trial"]:
        b, l = p["per_trial"]["stock"], p["per_trial"]["layer"]
        print("   per trial: stock %.1f turns, $%.4f, %d timeouts in %d  ->  layer %.1f turns, $%.4f, %d timeouts in %d"
              % (b["turns"], b["usd"], b["timeouts"], b["trials"], l["turns"], l["usd"], l["timeouts"], l["trials"]))


def grade(job_dir):
    """Each writer on the tasks whose reference solution passes the hidden tests: would its checks accept
    that correct solution, and reject the untouched starting state?"""
    graded, skipped, per = [], [], {}
    for d in sorted(p for p in Path(job_dir).iterdir() if p.is_dir() and "__" in p.name):
        f = d / "result.json"
        if not f.exists() or f.stat().st_size == 0:
            continue
        r = json.load(f.open(encoding="utf-8"))
        task = d.name.split("__")[0]
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        ce = ((r.get("agent_result") or {}).get("metadata") or {}).get("checkeval")
        if reward is None or reward < 1 or not ce:
            skipped.append("%s (%s)" % (task, "reference solution failed" if reward is not None else
                                        (r.get("exception_info") or {}).get("exception_type") or "no verdict"))
            continue
        graded.append(task)
        for name, rec in ce["writers"].items():
            w = per.setdefault(name, {"model": rec["writer"].get("model"), "cost": 0.0, "tasks": []})
            tok = rec["writer"].get("tokens") or [0, 0]
            pi, po = PRICES.get(w["model"], (0, 0))
            w["cost"] += (tok[0] * pi + tok[1] * po) / 1e6
            pairs = list(zip(rec.get("start", []), rec.get("solved", [])))
            usable = lambda s: s["status"] in ("pass", "fail")
            solved_usable = [b for _, b in pairs if usable(b)]
            w["tasks"].append({
                "task": task,
                "wrote": bool(solved_usable),
                "accepts_correct": bool(solved_usable) and all(b["status"] == "pass" for b in solved_usable),
                "rejects_untouched": any(usable(a) and a["status"] == "fail" for a, _ in pairs),
                "ideal": sum(1 for a, b in pairs if a["status"] == "fail" and b["status"] == "pass"),
                "wrong_on_correct": sum(1 for _, b in pairs if b["status"] == "fail"),
                "vacuous": sum(1 for a, b in pairs if a["status"] == "pass" and b["status"] == "pass"),
                "broken": sum(1 for _, b in pairs if not usable(b)),
                "checks": len(pairs),
            })
    writers = {}
    for name, w in per.items():
        ts, n = w["tasks"], len(w["tasks"])
        checks = sum(t["checks"] for t in ts)
        writers[name] = {
            "model": w["model"], "tasks": n, "checks": checks, "usd_per_task": w["cost"] / max(n, 1),
            "wrote": sum(t["wrote"] for t in ts) / max(n, 1),
            "accepts_correct": sum(t["accepts_correct"] for t in ts) / max(n, 1),
            "rejects_untouched": sum(t["rejects_untouched"] for t in ts) / max(n, 1),
            "both": sum(1 for t in ts if t["accepts_correct"] and t["rejects_untouched"]) / max(n, 1),
            "ideal": sum(t["ideal"] for t in ts) / max(checks, 1),
            "wrong": sum(t["wrong_on_correct"] for t in ts) / max(checks, 1),
            "vacuous": sum(t["vacuous"] for t in ts) / max(checks, 1),
            "broken": sum(t["broken"] for t in ts) / max(checks, 1),
            "per_task": ts,
        }
    return {"job": Path(job_dir).name, "graded": graded, "skipped": skipped, "writers": writers}


def checkeval(args):
    g = grade(args.job)
    print("## %s: %d tasks graded (reference solution passes the hidden tests)" % (g["job"], len(g["graded"])))
    if g["skipped"]:
        print("   not graded: %s" % ", ".join(g["skipped"]))
    print("\n%-8s %-35s %7s %9s %10s %6s  %6s %7s %7s %7s %7s  %8s" % (
        "writer", "model", "wrote", "accepts", "rejects", "both", "checks", "ideal", "wrong", "vacuous", "broken", "$/task"))
    print("%-8s %-35s %7s %9s %10s %6s  %6s %7s %7s %7s %7s" % ("", "", "checks", "correct", "untouched", "", "", "f->p", "->fail", "p->p", ""))
    pct = lambda v: "%d%%" % round(100.0 * v)
    for name, w in g["writers"].items():
        print("%-8s %-35s %7s %9s %10s %6s  %6d %7s %7s %7s %7s  %8.4f" % (
            name, (w["model"] or "")[:35], pct(w["wrote"]), pct(w["accepts_correct"]), pct(w["rejects_untouched"]),
            pct(w["both"]), w["checks"], pct(w["ideal"]), pct(w["wrong"]), pct(w["vacuous"]), pct(w["broken"]), w["usd_per_task"]))
    if args.table:
        names = list(g["writers"])
        print("\n%-32s " % "task" + " ".join("%-12s" % n for n in names) + "   (A = accepts the correct solution, R = rejects the start)")
        for i, task in enumerate(g["graded"]):
            cells = []
            for n in names:
                t = g["writers"][n]["per_task"][i]
                cells.append("%-12s" % ("%s%s %d/%d" % ("A" if t["accepts_correct"] else "-", "R" if t["rejects_untouched"] else "-",
                                                        t["wrong_on_correct"], t["checks"])))
            print("%-32s " % task[:32] + " ".join(cells))


RULES = [("any", lambda f, n: f >= 1), ("1/3", lambda f, n: f / n >= 1 / 3), ("1/2", lambda f, n: f / n >= 0.5),
         ("2/3", lambda f, n: f / n >= 2 / 3), ("all", lambda f, n: f == n)]


def thresholds(eval_job, dev_job, writer="nano-v1"):
    """'Reject when at least a share k of the usable checks fail', scored both ways: on the correct
    reference solutions and untouched starts of a checkeval job, and on the final states of a layered
    dev job, labelled by the hidden tests."""
    correct, start, wrong, right = [], [], [], []
    for d in sorted(p for p in Path(eval_job).iterdir() if p.is_dir() and "__" in p.name):
        f = d / "result.json"
        if not f.exists() or f.stat().st_size == 0:
            continue
        r = json.load(f.open(encoding="utf-8"))
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        rec = (((r.get("agent_result") or {}).get("metadata") or {}).get("checkeval") or {}).get("writers", {}).get(writer)
        if reward is None or reward < 1 or not rec:
            continue
        for state, bucket in (("solved", correct), ("start", start)):
            u = [s for s in rec.get(state, []) if s["status"] in ("pass", "fail")]
            if u:
                bucket.append((sum(s["status"] == "fail" for s in u), len(u)))
    for r in trials(dev_job):
        if not r["checks"] or not r["checks"]["rounds"]:
            continue
        u = [c for c in r["checks"]["rounds"][-1]["results"] if c["status"] in ("pass", "fail")]
        if u:
            (right if r["passed"] else wrong).append((sum(c["status"] == "fail" for c in u), len(u)))
    rows = []
    for name, rule in RULES:
        rows.append({"rule": name,
                     "accepts_correct": [sum(1 for f, n in correct if not rule(f, n)), len(correct)],
                     "rejects_untouched": [sum(1 for f, n in start if rule(f, n)), len(start)],
                     "rejects_wrong": [sum(1 for f, n in wrong if rule(f, n)), len(wrong)],
                     "accepts_right": [sum(1 for f, n in right if not rule(f, n)), len(right)]})
    return {"writer": writer, "rows": rows}


def print_thresholds(args):
    t = thresholds(args.eval_job, args.dev_job, args.writer)
    pct = lambda p: "%d/%d (%d%%)" % (p[0], p[1], round(100.0 * p[0] / p[1])) if p[1] else "-"
    print("writer %s: reject when at least this share of usable checks fail" % t["writer"])
    print("%-5s %22s %22s %30s %30s" % ("rule", "accepts correct soln", "rejects untouched", "rejects Nano's wrong final", "accepts Nano's right final"))
    for row in t["rows"]:
        print("%-5s %22s %22s %30s %30s" % (row["rule"], pct(row["accepts_correct"]), pct(row["rejects_untouched"]),
                                            pct(row["rejects_wrong"]), pct(row["accepts_right"])))


def misfiled(job_dir):
    """Stock-agent turns that came back with empty content and the JSON answer in reasoning_content,
    against the turns that came back normally."""
    empty, normal, total = [], [], 0
    for d in sorted(p for p in Path(job_dir).iterdir() if p.is_dir() and "__" in p.name):
        tj = d / "agent" / "trajectory.json"
        if not tj.exists():
            continue
        for s in json.load(tj.open(encoding="utf-8", errors="replace")).get("steps", []):
            if s.get("source") != "agent":
                continue
            total += 1
            content, reasoning = (s.get("message") or "").strip(), (s.get("reasoning_content") or "").lstrip()
            rec = ((s.get("metrics") or {}).get("completion_tokens") or 0, reasoning[:1] == "{" or reasoning[:3] == "```")
            if not content and '"commands"' in reasoning:
                empty.append(rec)
            elif content:
                normal.append(rec)
    median = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
    return {"turns": total, "misfiled": len(empty), "normal": len(normal),
            "median_tokens": [median([t for t, _ in empty]), median([t for t, _ in normal])],
            "opens_with_answer": [sum(o for _, o in empty), sum(o for _, o in normal)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    e = sub.add_parser("checkeval")
    e.add_argument("job", help="job dir of a tally.checkeval run")
    e.add_argument("--table", action="store_true", help="one row per task")
    c = sub.add_parser("compare")
    c.add_argument("job", help="job dir of the layered agent")
    c.add_argument("--table", action="store_true", help="one row per task first")
    t = sub.add_parser("thresholds")
    t.add_argument("eval_job")
    t.add_argument("dev_job")
    t.add_argument("--writer", default="nano-v1")
    m = sub.add_parser("misfiled")
    m.add_argument("job", nargs="?", default=str(ROOT / "jobs" / BASELINE))
    args = ap.parse_args()
    if args.cmd == "misfiled":
        r = misfiled(args.job)
        print("%d agent turns: %d came back with empty content and the answer in reasoning_content (%.0f%%), %d normally"
              % (r["turns"], r["misfiled"], 100.0 * r["misfiled"] / max(r["turns"], 1), r["normal"]))
        print("median completion tokens: misfiled %d, normal %d; 'reasoning' opens with { or a fence: misfiled %d, normal %d"
              % (r["median_tokens"][0], r["median_tokens"][1], r["opens_with_answer"][0], r["opens_with_answer"][1]))
        return
    {"split": split, "compare": compare, "checkeval": checkeval, "thresholds": print_thresholds}[args.cmd](args)


if __name__ == "__main__":
    main()
