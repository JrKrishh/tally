"""Build docs/data.js for the demo page from local results.

  python -m tally.site

Everything on the page comes from here: the planner grid is select.simulate run
over every control combination, the real run is read from the Harbor job, and
the receipt is priced from the job's own token stats.
"""
import collections
import datetime
import json
import random
import sys
from pathlib import Path

import numpy as np

from . import select
from .pull import DATA

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
JOB = "tally-terminalbench-run-nvidia-NVIDIA-Nemotron-3-Nano-30B-A3B-20260912-165615"
VALIDATE_PREFIX = "tally-terminalbench-validate-nvidia-NVIDIA-Nemotron-3-Nano-30B-A3B-"
LAYER_DEV_JOB = "tally-terminalbench-dev-CheckedTerminus-nvidia-NVIDIA-Nemotron-3-Nano-30B-A3B-20260913-143224"
LAYER_HELDOUT_JOB = "tally-terminalbench-heldout-CheckedTerminus-nvidia-NVIDIA-Nemotron-3-Nano-30B-A3B-20260913-154405"
CHECKEVAL_JOB = "tally-terminalbench-dev-CheckEval-nvidia-NVIDIA-Nemotron-3-Nano-30B-A3B-20260913-181951"
WRITER_NAMES = {"nano-v1": ("Nemotron 3 Nano 30B", "original"), "nano": ("Nemotron 3 Nano 30B", "stricter"),
                "super": ("Nemotron 3 Super 120B", "stricter"), "ultra": ("Nemotron 3 Ultra 550B", "stricter")}
THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
ATTEMPTS = [1, 2, 3, 5, None]
PRICE_IN, PRICE_OUT, VM_PER_HOUR = 0.06e-6, 0.24e-6, 0.23
NAMES = {
    "openai/gpt-5.4-2026-03-05": "GPT-5.4",
    "openai/gpt-5.2-2025-12-11": "GPT-5.2",
    "openai/gpt-5-2025-08-07": "GPT-5",
    "anthropic/claude-opus-4-6": "Opus 4.6",
    "anthropic/claude-opus-4-5-20251101": "Opus 4.5",
    "anthropic/claude-opus-4-20250514": "Opus 4",
}


def planner(col):
    by, models, common = select.cells(select.load_attempts(col))
    acc, cost = select.full(by, models, common)
    total = sum(cost.values())
    grid = {}
    for mode in ("both", "fail"):
        for t in THRESHOLDS:
            lo, hi = t, (1.0 - t if mode == "both" else 1.0)
            for k in ATTEMPTS:
                seeds = select.SEEDS if k else 1
                est_sum = collections.defaultdict(float)
                cf = sp = mae = top = nrun = 0.0
                for seed in range(seeds):
                    est, spent, n = select.simulate(by, models, common, lo, hi, k, random.Random(seed))
                    for m in models:
                        est_sum[m] += est[m]
                    cf += sum(spent.values()) / total
                    sp += select.spearman([acc[m] for m in models], [est[m] for m in models])
                    mae += float(np.mean([abs(est[m] - acc[m]) for m in models]))
                    top += max(models, key=lambda m: acc[m]) == max(models, key=lambda m: est[m])
                    nrun += n
                grid["%s|%.2f|%s" % (mode, t, k or "all")] = {
                    "cost": round(cf / seeds, 4), "spearman": round(sp / seeds, 3), "mae": round(mae / seeds, 3),
                    "top1": [int(top), seeds], "tasks_run": round(nrun / seeds / len(common), 3),
                    "est": dict((NAMES[m], round(est_sum[m] / seeds, 3)) for m in models),
                }
        print("  %s %s: %d cells" % (col, mode, len(grid)))
    return {
        "tasks": len(common), "tokens_full": total,
        "full": dict((NAMES[m], round(acc[m], 3)) for m in models),
        "grid": grid,
    }


def trial_rows(job_dir):
    rows = []
    for d in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        f = d / "result.json"
        if not f.exists() or f.stat().st_size == 0:
            continue
        r = json.load(f.open(encoding="utf-8"))
        et = (r.get("exception_info") or {}).get("exception_type")
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if reward is None:
            continue
        steps, tj = 0, d / "agent" / "trajectory.json"
        if tj.exists():
            steps = len(json.load(tj.open(encoding="utf-8", errors="replace")).get("steps", []))
        ae = r.get("agent_execution") or {}
        parse = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        mins = (parse(ae["finished_at"]) - parse(ae["started_at"])).total_seconds() / 60 if ae.get("finished_at") else 0.0
        ar = r.get("agent_result") or {}
        rows.append({
            "task": d.name.split("__")[0], "id": d.name.split("__")[1],
            "outcome": "timeout" if et == "AgentTimeoutError" else ("pass" if reward >= 1.0 else "fail"),
            "steps": steps, "minutes": round(mins, 1),
            "tokens": (ar.get("n_input_tokens") or 0) + (ar.get("n_output_tokens") or 0),
            "started": r.get("started_at"), "finished": r.get("finished_at"),
        })
    return rows


def estimate(by, skipped, tasks):
    """Mean accuracy over tasks (measured where run, history where skipped) and a 95% interval
    from bootstrapping tasks and, within each measured task, its attempts."""
    score = lambda t: float(np.mean([1.0 if a["outcome"] == "pass" else 0.0 for a in by[t]])) if t in by else skipped[t]
    point = float(np.mean([score(t) for t in tasks]))
    rng, boots = random.Random(0), []
    for _ in range(4000):
        s = []
        for t in (rng.choice(tasks) for _ in tasks):
            if t in by:
                a = [1.0 if x["outcome"] == "pass" else 0.0 for x in by[t]]
                s.append(sum(rng.choice(a) for _ in a) / len(a))
            else:
                s.append(skipped[t])
        boots.append(sum(s) / len(s))
    boots.sort()
    return point, [round(boots[100], 3), round(boots[3899], 3)]


def real_run():
    job_dir = ROOT / "jobs" / JOB
    rows = trial_rows(job_dir)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    for t in by:
        by[t].sort(key=lambda r: r["started"])
    plan = json.load(open(DATA.parent / "plan_terminalbench.json"))
    skipped = plan["skipped_prior"]
    tasks = sorted(set(by) | set(skipped))
    est, ci = estimate(by, skipped, tasks)
    ref, ref_tasks, excluded = select.nofeedback_reference("terminalbench")
    ref_tasks = [t for t in ref_tasks if t in by or t in skipped]
    cmp_est, cmp_ci = estimate(by, skipped, ref_tasks)
    solved = sorted((t for t, v in by.items() if any(a["outcome"] == "pass" for a in v)),
                    key=lambda t: (-sum(a["outcome"] == "pass" for a in by[t]), t))
    stats = json.load(open(job_dir / "result.json")).get("stats", {})
    tin, tout, tcache = stats["n_input_tokens"], stats["n_output_tokens"], stats["n_cache_tokens"]
    # wall-clock of the two sittings: split sorted trial times wherever the VM sat idle for over an hour
    spans, times = [], sorted((r["started"], r["finished"]) for r in rows)
    parse = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    start, end = parse(times[0][0]), parse(times[0][1])
    for s, f in times[1:]:
        s, f = parse(s), parse(f)
        if (s - end).total_seconds() > 3600:
            spans.append((start, end)); start = s
        end = max(end, f)
    spans.append((start, end))
    hours = sum((b - a).total_seconds() for a, b in spans) / 3600
    always_fail = sum(1 for v in by.values() if all(a["outcome"] != "pass" for a in v))
    return {
        "model": "Nemotron 3 Nano 30B", "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "estimate": round(est, 3), "ci": ci,
        "tasks_all": len(tasks), "trials": len(rows), "passes": sum(r["outcome"] == "pass" for r in rows),
        "timeouts": sum(r["outcome"] == "timeout" for r in rows),
        "tasks_run": len(by), "always_fail": always_fail,
        "always_pass": [t for t in solved if all(a["outcome"] == "pass" for a in by[t])],
        "solved": [{"task": t, "attempts": [dict((k, a[k]) for k in ("id", "outcome", "steps", "minutes", "tokens")) for a in by[t]]} for t in solved],
        "compare": {
            "tasks": len(ref_tasks), "nano": round(cmp_est, 3), "ci": cmp_ci,
            "frontier": dict((NAMES[m], round(v, 3)) for m, v in ref.items()),
            "excluded": dict((NAMES[m], n) for m, n in excluded.items()),
        },
        "receipt": {
            "tokens_in": tin, "tokens_out": tout, "tokens_cached": tcache,
            "price_in_per_m": PRICE_IN * 1e6, "price_out_per_m": PRICE_OUT * 1e6,
            "inference_usd": round(tin * PRICE_IN + tout * PRICE_OUT, 2),
            "vm_hours": round(hours, 1), "vm_per_hour": VM_PER_HOUR, "vm_usd": round(hours * VM_PER_HOUR, 2),
            "sittings": [[a.strftime("%H:%M"), b.strftime("%H:%M")] for a, b in spans],
        },
    }


def validation():
    jobs = sorted((ROOT / "jobs").glob(VALIDATE_PREFIX + "*"), key=lambda p: p.stat().st_mtime)
    jobs = [j for j in jobs if not j.name.endswith("-dry")]
    if not jobs:
        return []
    priors = json.load(open(DATA.parent / "plan_terminalbench_v1_both_sides.json"))["skipped_prior"]
    out = []
    for r in trial_rows(jobs[-1]):
        p = priors.get(r["task"])
        if p is None:
            continue
        out.append({"task": r["task"], "prior": round(p, 2), "predicted": "pass" if p > 0.5 else "fail", "actual": r["outcome"]})
    return sorted(out, key=lambda x: (x["predicted"], x["task"]))


def cold_start():
    from scipy.stats import spearmanr
    out = {}
    for col in ("terminalbench", "swebenchpro"):
        tasks = dict((json.loads(l)["task"], json.loads(l)) for l in open(DATA.parent / ("tasks_%s.jsonl" % col), encoding="utf-8") if l.strip())
        sc = {}
        for l in open(DATA.parent / ("coldstart_%s_nvidia-nemotron-3-super-120b-a12b.jsonl" % col), encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("p_solve") is not None:
                    sc[r["task"]] = r["p_solve"]
        keys = [t for t in sc if t in tasks]
        out[col] = {
            "tasks": len(keys),
            "model": round(float(spearmanr([tasks[t]["difficulty"] for t in keys], [sc[t] for t in keys]).correlation), 3),
            "length": round(float(spearmanr([tasks[t]["difficulty"] for t in keys], [tasks[t]["chars"] for t in keys]).correlation), 3),
        }
    return out


def layer():
    """The verify-before-done layer experiment: dev and held-out paired against the stock agent, the
    check-writer grade, and the misfiled answers that motivated half of it."""
    from . import boost
    r3 = lambda v: round(v, 3)

    def run(job):
        p = boost.paired(ROOT / "jobs" / job)
        s, l = p["per_trial"]["stock"], p["per_trial"]["layer"]
        return {
            "tasks": p["tasks"], "trials": p["trials"], "attempts": round(p["trials"] / p["tasks"]),
            "stock": r3(p["stock"]), "layer": r3(p["layer"]), "diff": r3(p["diff"]), "ci": [r3(c) for c in p["ci"]],
            "stock_passes": p["stock_passes"], "layer_passes": p["layer_passes"],
            "gained": p["gained"], "lost": p["lost"], "false_claims": p["false_claims"],
            "rejected": p["decisions"].get("rejected", 0), "verdicts": p["verdicts"],
            "recovered_turns": p["recovered_turns"],
            "per_trial": {"stock": {"turns": round(s["turns"], 1), "usd": round(s["usd"], 4), "timeouts": s["timeouts"], "trials": s["trials"]},
                          "layer": {"turns": round(l["turns"], 1), "usd": round(l["usd"], 4), "timeouts": l["timeouts"], "trials": l["trials"]}},
            "inference_usd": round(l["tokens"][0] * PRICE_IN + l["tokens"][1] * PRICE_OUT, 2),
        }

    g = boost.grade(ROOT / "jobs" / CHECKEVAL_JOB)
    writers = []
    for key, w in g["writers"].items():
        model, prompt = WRITER_NAMES[key]
        writers.append({"model": model, "prompt": prompt, "accepts_correct": r3(w["accepts_correct"]),
                        "rejects_untouched": r3(w["rejects_untouched"]), "wrong": r3(w["wrong"]),
                        "usd_per_task": round(w["usd_per_task"], 4), "checks": w["checks"]})
    th = boost.thresholds(ROOT / "jobs" / CHECKEVAL_JOB, ROOT / "jobs" / LAYER_DEV_JOB)
    m = boost.misfiled(ROOT / "jobs" / JOB)
    return {
        "dev": run(LAYER_DEV_JOB), "heldout": run(LAYER_HELDOUT_JOB),
        "writers": {"graded": len(g["graded"]), "rows": writers}, "thresholds": th["rows"],
        "misfiled": {"turns": m["turns"], "misfiled": m["misfiled"], "median_tokens": m["median_tokens"],
                     "opens_with_answer": m["opens_with_answer"], "normal": m["normal"]},
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tb, swe = planner("terminalbench"), planner("swebenchpro")
    data = {
        "planner": {"terminalbench": tb, "swebenchpro": swe, "thresholds": THRESHOLDS, "attempts": [a or "all" for a in ATTEMPTS]},
        "run": real_run(),
        "validation": validation(),
        "coldstart": cold_start(),
        "layer": layer(),
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.js").write_text("window.TALLY = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    r = data["run"]
    print("wrote %s  (%d KB)" % (DOCS / "data.js", (DOCS / "data.js").stat().st_size // 1024))
    print("  run: estimate %.3f [%.3f, %.3f], %d trials, %d passes, solved tasks %d, receipt $%.2f + VM $%.2f (%.1f h: %s)"
          % (r["estimate"], r["ci"][0], r["ci"][1], r["trials"], r["passes"], len(r["solved"]),
             r["receipt"]["inference_usd"], r["receipt"]["vm_usd"], r["receipt"]["vm_hours"], r["receipt"]["sittings"]))
    c = r["compare"]
    print("  like for like, %d tasks, no feedback: %s | Nano %.3f %s | excluded %s"
          % (c["tasks"], c["frontier"], c["nano"], c["ci"], c["excluded"]))
    print("  validation:", [(v["task"], v["predicted"], v["actual"]) for v in data["validation"]])
    print("  coldstart:", data["coldstart"])
    L = data["layer"]
    for half in ("dev", "heldout"):
        h = L[half]
        print("  layer %s: %d tasks x %d, stock %.3f -> layer %.3f, diff %+.3f %s, inference $%.2f"
              % (half, h["tasks"], h["attempts"], h["stock"], h["layer"], h["diff"], h["ci"], h["inference_usd"]))
    print("  writers:", [(w["model"], w["prompt"], w["accepts_correct"]) for w in L["writers"]["rows"]], "| misfiled:", L["misfiled"])
    print("  headline cell both|0.10|3:", tb["grid"]["both|0.10|3"])


if __name__ == "__main__":
    main()
