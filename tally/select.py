"""Step 3: how much does cell selection save, and does the ranking survive?

Step 2 showed a new model's per-task outcome is predictable from the other
models' pass rates before its first token. So: for each model in turn, pretend
it is new, take task difficulty from the other five, give the tasks whose
difficulty is uncertain their attempts, and either impute the certain ones from
the prior or run them once, and see (a) what fraction of the tokens were spent
and (b) whether the six-model ranking and each model's accuracy survive. Every
number is on the tasks all six share, averaged over SEEDS attempt samples. The
"every task" rows (window [0, 1]) are the baseline that needs no history at all.

  python -m tally.select

AUC was never a dollar figure. This is.
"""
import argparse
import collections
import csv
import random
import sys

import numpy as np

from .pull import COLLECTIONS, DATA
from .matrix import passed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WINDOWS = [(0.0, 1.0), (0.05, 0.95), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
ATTEMPTS = [None, 3, 2, 1]     # attempts per uncertain cell: all, or capped
CERTAIN = [0, 1]               # attempts per certain cell: 0 imputes the prior, 1 measures it once
SEEDS = 200                    # at 5 seeds Terminal-Bench's skip+cap-3 row read 0.989; at 200 it reads 0.973


def load_attempts(col):
    p = DATA.parent / ("attempts_%s.csv" % col)
    if not p.exists():
        sys.exit("no %s -- run: python -m tally.matrix" % p)
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    seen, out = set(), []                      # cross-shard duplicate deposits
    for r in rows:
        k = (r["model"], r["task"], r["score"], r["turns"], r["total_tokens"], r["latency_ms"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def cells(rows):
    """by[(model, task)] = [(passed, tokens), ...]; plus models and the tasks all share."""
    by = collections.defaultdict(list)
    for r in rows:
        s = float(r["score"]) if r["score"] not in ("", "None") else None
        tok = float(r["total_tokens"]) if r["total_tokens"] not in ("", "None") else 0.0
        by[(r["model"], r["task"])].append((1.0 if passed(s) else 0.0, tok))
    models = sorted(set(m for m, _ in by))
    tasks = sorted(set(t for _, t in by))
    common = [t for t in tasks if all((m, t) in by for m in models)]
    return by, models, common


def nofeedback_reference(col, min_tasks=20):
    """History to rank a new model against: each model's accuracy from attempts WITHOUT
    correctness feedback (the study's S-adaptive arm), on the tasks shared by every model
    with at least min_tasks such tasks. The other arm told the agent whether its answer was
    right, which roughly doubles the weaker models - a signal a new model never gets.
    -> (acc by model, common task list, {excluded model: no-feedback task count})."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load_attempts(col):
        if r["strategy"] != "S-adaptive":
            continue
        s = float(r["score"]) if r["score"] not in ("", "None") else None
        per[r["model"]][r["task"]].append(1.0 if passed(s) else 0.0)
    counts = dict((m, len(v)) for m, v in per.items())
    included = [m for m in per if counts[m] >= min_tasks]
    common = sorted(set.intersection(*(set(per[m]) for m in included))) if included else []
    acc = dict((m, float(np.mean([np.mean(per[m][t]) for t in common]))) for m in included)
    excluded = dict((m, n) for m, n in counts.items() if m not in included)
    return acc, common, excluded


def full(by, models, common):
    acc = dict((m, float(np.mean([np.mean([p for p, _ in by[(m, t)]]) for t in common]))) for m in models)
    cost = dict((m, sum(tok for t in common for _, tok in by[(m, t)])) for m in models)
    return acc, cost


def simulate(by, models, common, lo, hi, k, rng, diff_fn=None, k_certain=0):
    """Each model treated as new in turn. -> estimated accuracy, tokens spent, uncertain tasks.
    diff_fn(model, task) replaces the history prior, e.g. with a cold-start estimate.
    k_certain > 0 runs the tasks outside the window that many times instead of imputing them."""
    est, spent, nrun = {}, {}, []
    for m in models:
        others = [o for o in models if o != m]
        if diff_fn:
            diff = dict((t, float(diff_fn(m, t))) for t in common)
        else:
            diff = dict((t, float(np.mean([np.mean([p for p, _ in by[(o, t)]]) for o in others]))) for t in common)
        run = set(t for t in common if lo <= diff[t] <= hi)
        nrun.append(len(run))
        a, c = 0.0, 0.0
        for t in common:
            if t in run or k_certain:
                atts = list(by[(m, t)])
                rng.shuffle(atts)
                cap = k if t in run else k_certain
                if cap:
                    atts = atts[:cap]
                a += float(np.mean([p for p, _ in atts]))
                c += sum(tok for _, tok in atts)
            else:
                a += diff[t]                   # impute from the prior, spend nothing
        est[m] = a / len(common)
        spent[m] = c
    return est, spent, float(np.mean(nrun))


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    for col in COLLECTIONS:
        by, models, common = cells(load_attempts(col))
        acc, cost = full(by, models, common)
        total = sum(cost.values())
        print("\n## %s: %d models, %d shared tasks, %.1fM tokens for the full evaluation" % (col, len(models), len(common), total / 1e6))
        print("  full ranking:")
        for m in sorted(models, key=lambda m: -acc[m]):
            print("    %-38s acc %.3f   %.1fM tokens" % (m, acc[m], cost[m] / 1e6))
        print("\n  %-14s %-9s %-8s %9s %8s %10s %9s %s" % ("difficulty", "attempts", "certain", "uncertain", "cost", "spearman", "acc MAE", "top-1 kept"))
        for lo, hi in WINDOWS:
            for kc in CERTAIN:
                if kc and (lo, hi) == (0.0, 1.0):
                    continue                   # every task is already uncertain
                for k in ATTEMPTS:
                    sp, mae, cf, nr, top = [], [], [], [], []
                    for seed in range(SEEDS if (k or kc) else 1):
                        est, spent, n = simulate(by, models, common, lo, hi, k, random.Random(seed), k_certain=kc)
                        order_true = sorted(models, key=lambda m: -acc[m])
                        order_est = sorted(models, key=lambda m: -est[m])
                        sp.append(spearman([acc[m] for m in models], [est[m] for m in models]))
                        mae.append(float(np.mean([abs(est[m] - acc[m]) for m in models])))
                        cf.append(sum(spent.values()) / total)
                        nr.append(n)
                        top.append(order_true[0] == order_est[0])
                    print("  [%.2f, %.2f]   %-9s %-8s %8.0f%% %7.1f%% %10.3f %9.3f   %5.1f%%"
                          % (lo, hi, "all" if k is None else "<=%d" % k, "run once" if kc else "impute",
                             100.0 * np.mean(nr) / len(common), 100.0 * np.mean(cf), np.mean(sp), np.mean(mae),
                             100.0 * np.mean(top)))


if __name__ == "__main__":
    main()
