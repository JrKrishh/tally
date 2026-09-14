"""tally plan: from history, decide which cells to run for a NEW model.

Step 3 showed which cells are worth running. This turns that into a concrete
plan for a model nobody has evaluated: the tasks whose difficulty is uncertain
get run with a capped number of attempts, and the tasks history calls certain
are run once each (the validate phase), so every task is measured. Step 3's
replay puts that at 14.7% of Terminal-Bench's tokens, Spearman 0.982, within
0.004 of running every task twice for 16% fewer tokens. `--validate N` runs
only N of the certain tasks and imputes the rest from history, the plan the
first real run used. The plan is a JSON file that `tally run` executes and
`tally report` scores.

  python -m tally.plan --benchmark terminalbench
  python -m tally.plan --benchmark terminalbench --lo 0.10 --hi 1.0 --attempts 3 --validate 8

Expected token cost is estimated from the frontier models' attempts on each
task. A different model will spend differently; the number is a scale, not a
quote.
"""
import argparse
import collections
import json
import random
import statistics
import sys

import numpy as np

from . import select
from .pull import DATA

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASETS = {"terminalbench": "terminal-bench@2.0", "swebenchpro": "swe-bench-pro"}


def history(col, min_models=3):
    """-> difficulty[task], median tokens per attempt[task], models seen[task]."""
    by, models, _ = select.cells(select.load_attempts(col))
    per_task = collections.defaultdict(dict)
    toks = collections.defaultdict(list)
    for (m, t), atts in by.items():
        per_task[t][m] = float(np.mean([p for p, _ in atts]))
        toks[t] += [tok for _, tok in atts if tok]
    diff, med, seen = {}, {}, {}
    for t, rates in per_task.items():
        if len(rates) >= min_models:
            diff[t] = float(np.mean(list(rates.values())))
            med[t] = float(statistics.median(toks[t])) if toks[t] else 0.0
            seen[t] = len(rates)
    return diff, med, seen, models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="terminalbench", choices=list(DATASETS))
    ap.add_argument("--lo", type=float, default=0.10)
    ap.add_argument("--hi", type=float, default=0.90)
    ap.add_argument("--attempts", type=int, default=2, help="attempts per uncertain task")
    ap.add_argument("--validate", default="all",
                    help="certain tasks to run once: 'all' measures every one, N runs a sample and imputes the rest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    diff, med, seen, models = history(args.benchmark)
    all_ids = sorted(diff)
    ids_file = DATA.parent / ("%s_task_ids.json" % {"terminalbench": "tb2"}.get(args.benchmark, args.benchmark))
    no_history = []
    if ids_file.exists():
        bench_ids = json.load(ids_file.open())
        no_history = sorted(t for t in bench_ids if t not in diff)
        all_ids = sorted(set(all_ids) | set(bench_ids))

    run = sorted(t for t in diff if args.lo <= diff[t] <= args.hi)
    skipped = sorted(t for t in diff if t not in set(run))
    rng = random.Random(args.seed)
    n_validate = len(skipped) if args.validate == "all" else int(args.validate)
    validate = sorted(rng.sample(skipped, min(n_validate, len(skipped))))

    med_all = statistics.median(med.values()) if med else 0.0
    cost_plan = (sum(med[t] for t in run) * args.attempts
                 + sum(med[t] for t in validate)
                 + len(no_history) * med_all * args.attempts)
    cost_full = sum(med.get(t, med_all) for t in all_ids) * args.attempts
    cost_study = sum(med.get(t, med_all) * seen.get(t, 0) * 10 for t in all_ids) / max(1, len(models))

    plan = {
        "benchmark": args.benchmark, "dataset": DATASETS[args.benchmark],
        "window": [args.lo, args.hi], "attempts": args.attempts, "seed": args.seed,
        "history_models": models,
        "tasks_run": run,
        "tasks_validate": validate,
        "tasks_no_history": no_history,
        "skipped_prior": dict((t, round(diff[t], 3)) for t in skipped),
        "run_prior": dict((t, round(diff[t], 3)) for t in run),     # so report can impute cells not yet measured
        "expected_tokens": {"plan": cost_plan, "full_at_same_attempts": cost_full},
    }
    out = args.out or str(DATA.parent / ("plan_%s.json" % args.benchmark))
    json.dump(plan, open(out, "w"), indent=1)

    print("## plan for a new model on %s  (%s)" % (args.benchmark, DATASETS[args.benchmark]))
    print("  history: %d models, %d tasks with >=3 models' attempts" % (len(models), len(diff)))
    print("  run     : %3d tasks in difficulty [%.2f, %.2f], %d attempts each" % (len(run), args.lo, args.hi, args.attempts))
    print("  certain : %3d tasks history calls certain  (%d near-fail, %d near-pass)"
          % (len(skipped), sum(1 for t in skipped if diff[t] < args.lo), sum(1 for t in skipped if diff[t] > args.hi)))
    if len(validate) == len(skipped):
        print("  validate: all %d run once, so none is imputed" % len(validate))
    else:
        print("  validate: %3d of them run once as a check, the rest imputed: %s" % (len(validate), ", ".join(validate)))
    if no_history:
        print("  no history, run anyway: %s" % ", ".join(no_history))
    print("  expected tokens: plan %.0fM   vs %.0fM for every task at %d attempts   (%.0f%%)"
          % (cost_plan / 1e6, cost_full / 1e6, args.attempts, 100.0 * cost_plan / cost_full if cost_full else 0))
    print("  (frontier-model token counts; a different model spends differently)")
    print("  wrote %s" % out)


if __name__ == "__main__":
    main()
