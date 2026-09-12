"""tally report: score a real run against its plan, with the receipt.

Reads Harbor's job results for the run and validate phases and produces what
an evaluation is for: the new model's estimated accuracy on the whole
benchmark (measured where it ran, imputed from history where it didn't), its
rank among the models history knows, the tokens it spent against what the
full evaluation would have cost, and -- the honest part -- whether the cells
we chose NOT to run came out the way history said they would.

  python -m tally.report --plan data/plan_terminalbench.json [--jobs-dir jobs] [--model ...]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import select
from .pull import DATA

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent


def find_tokens(obj, depth=0):
    """Sum any *token* counters found in an agent result, whatever Harbor nests them under."""
    total = 0
    if depth > 6:
        return 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and re.search(r"token", str(k), re.I) and not re.search(r"cache|id", str(k), re.I):
                total += v
            else:
                total += find_tokens(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            total += find_tokens(v, depth + 1)
    return total


def load_job(jobs_dir, job_name):
    """-> {task: [(reward, tokens, errored), ...]} from jobs/<job>/result.json, or per-trial files."""
    jd = Path(jobs_dir) / job_name
    if not jd.exists():
        return None
    trials = []
    top = jd / "result.json"
    if top.exists():
        j = json.load(top.open(encoding="utf-8"))
        trials = j.get("trial_results") or []
    if not trials:
        for f in sorted(jd.rglob("result.json")):
            if f == top:
                continue
            try:
                trials.append(json.load(f.open(encoding="utf-8")))
            except Exception:
                pass
    out = {}
    for t in trials:
        task = t.get("task_name") or t.get("task_id") or "?"
        rewards = ((t.get("verifier_result") or {}).get("rewards") or {})
        reward = rewards.get("reward") if isinstance(rewards, dict) else None
        errored = bool(t.get("exception_info"))
        out.setdefault(task, []).append((float(reward) if reward is not None else None, find_tokens(t.get("agent_result") or {}), errored))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--jobs-dir", default=str(ROOT / "jobs"))
    ap.add_argument("--model", default="nvidia/nemotron-3-nano-30b-a3b")
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-")
    lo, hi = plan["window"]

    run = load_job(args.jobs_dir, "tally-%s-run-%s" % (plan["benchmark"], slug)) or {}
    val = load_job(args.jobs_dir, "tally-%s-validate-%s" % (plan["benchmark"], slug)) or {}
    if not run and not val:
        sys.exit("no results under %s for %s -- run: python -m tally.run" % (args.jobs_dir, args.model))

    def rate(atts):
        r = [x for x, _, _ in atts if x is not None]
        return float(np.mean([1.0 if x >= 1.0 else 0.0 for x in r])) if r else None

    # measured cells
    measured = dict((t, rate(a)) for t, a in run.items() if rate(a) is not None)
    tokens_run = sum(tok for a in run.values() for _, tok, _ in a)
    errors = sum(1 for a in list(run.values()) + list(val.values()) for _, _, e in a if e)
    n_attempts = sum(len(a) for a in run.values())

    # estimate over the whole benchmark: measured where run, prior where skipped
    prior = plan["skipped_prior"]
    all_tasks = sorted(set(plan["tasks_run"]) | set(prior) | set(plan.get("tasks_no_history", [])))
    est_parts, covered = [], 0
    for t in all_tasks:
        if t in measured:
            est_parts.append(measured[t]); covered += 1
        elif t in prior:
            est_parts.append(prior[t])
    est = float(np.mean(est_parts)) if est_parts else float("nan")

    print("## %s on %s (%s)" % (args.model, plan["benchmark"], plan["dataset"]))
    print("  run phase   : %d tasks measured, %d attempts, %d errored trials" % (len(measured), n_attempts, errors))
    print("  measured pass rate on run tasks: %.3f" % float(np.mean(list(measured.values()))) if measured else "  (nothing measured yet)")
    print("  estimated accuracy on all %d tasks (measured where run, history where skipped): %.3f" % (len(all_tasks), est))
    if tokens_run:
        print("  tokens spent (run phase): %.1fM   plan expected ~%.0fM at frontier-model lengths" % (tokens_run / 1e6, plan["expected_tokens"]["plan"] / 1e6))
        print("  tokens per attempt: %.0fK   -> full plan at this rate: ~%.0fM" % (tokens_run / 1e3 / max(1, n_attempts), tokens_run / max(1, n_attempts) * (len(plan["tasks_run"]) * plan["attempts"] + len(plan["tasks_validate"])) / 1e6))
    else:
        print("  tokens: not found in agent results (inspect jobs/<job>/**/result.json)")

    # rank against the models history knows
    by, models, common = select.cells(select.load_attempts(plan["benchmark"]))
    acc, _ = select.full(by, models, common)
    table = sorted([(a, m) for m, a in acc.items()] + [(est, args.model + "  <- new, estimated")], reverse=True)
    print("\n  ranking (history models on their %d shared tasks; new model estimated):" % len(common))
    for i, (a, m) in enumerate(table, 1):
        print("   %d. %-50s %.3f" % (i, m, a))

    # the honest part: did the skipped cells behave as history predicted?
    if val:
        print("\n  validation of skipped cells (history said certain; we ran them once anyway):")
        agree = 0
        for t, atts in sorted(val.items()):
            r = rate(atts)
            p = prior.get(t)
            if r is None or p is None:
                continue
            predicted = "pass" if p > hi else "fail"
            actual = "pass" if r >= 0.5 else "fail"
            agree += predicted == actual
            print("   %-34s history %.2f -> predicted %-4s   actual %-4s   %s" % (t[:34], p, predicted, actual, "ok" if predicted == actual else "MISS"))
        print("  agreement: %d/%d" % (agree, len(val)))


if __name__ == "__main__":
    main()
