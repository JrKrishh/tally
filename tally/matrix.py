"""Rebuild the model x task x attempts matrix two ways and diff them.

The aggregates (small) list each shard's sample_ids and encode epochs in the
evaluation_name ("terminalbench/S-adaptive/+8ep"), so the matrix looks inferable
without the samples. It is not: adaptive sampling makes +Nep a ceiling, and some
shards carry rows with no task list. The aggregates undercount attempts 2-3x on
every (model, task) pair. The samples (7.4 GB) are the only ground truth; the
aggregate path is kept as the cross-check that proved it.

  python -m tally.matrix                       # both sources, diff, consistency
  python -m tally.matrix --source aggregates   # works before the big pull finishes
  python -m tally.matrix --source samples

Writes data/attempts_<collection>.csv -- one row per attempt -- when samples are
read. That flat table is what every later step builds on.
"""
import argparse
import collections
import csv
import json
import re
import statistics
import sys

from .pull import COLLECTIONS, DATA, progress

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EP = re.compile(r"/([^/]+)/\+(\d+)ep")
ATTEMPT_FIELDS = ["collection", "model", "task", "shard", "strategy", "epochs",
                  "score", "is_correct", "turns", "tool_calls",
                  "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "latency_ms", "error"]


def passed(score):
    return score is not None and score >= 1.0


# ---------------------------------------------------------------- aggregates

def from_aggregates(col):
    """-> cover[model][task] = attempts, strategies Counter, unparsed shard count."""
    cover = collections.defaultdict(collections.Counter)
    strat, unparsed = collections.Counter(), 0
    for p in sorted((DATA / "data" / col).rglob("*.json")):
        if p.name.endswith("_samples.jsonl"):
            continue
        a = json.loads(p.read_text(encoding="utf-8"))
        model = a["model_info"]["id"]
        for er in a.get("evaluation_results", []):
            m = EP.search(er.get("evaluation_name", ""))
            if m:
                s, ep = m.group(1), int(m.group(2))
            else:
                s, ep, unparsed = "?", 1, unparsed + 1
            strat[s] += 1
            for t in (er.get("source_data") or {}).get("sample_ids") or []:
                cover[model][t] += ep
    return cover, strat, unparsed


# ---------------------------------------------------------------- samples

def shard_config(agg_path):
    """(strategy, max epochs) from a shard's aggregate JSON; ('?', None) if unparseable."""
    if not agg_path.exists():
        return "?", None
    a = json.loads(agg_path.read_text(encoding="utf-8"))
    for er in a.get("evaluation_results", []):
        m = EP.search(er.get("evaluation_name", ""))
        if m:
            return m.group(1), int(m.group(2))
    return "?", None


def from_samples(col):
    """-> cover[model][task] = attempts, plus the flat attempts list."""
    cover = collections.defaultdict(collections.Counter)
    rows = []
    files = sorted((DATA / "data" / col).rglob("*_samples.jsonl"))
    for i, p in enumerate(files):
        model = p.parent.parent.name + "/" + p.parent.name
        shard = p.name[:-len("_samples.jsonl")]
        strategy, epochs = shard_config(p.parent / (shard + ".json"))
        with p.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                ev = r.get("evaluation") or {}
                tu = r.get("token_usage") or {}
                pf = r.get("performance") or {}
                task = r["sample_id"]
                cover[model][task] += 1
                rows.append({
                    "collection": col, "model": model, "task": task,
                    "shard": shard[:8], "strategy": strategy, "epochs": epochs,
                    "score": ev.get("score"), "is_correct": ev.get("is_correct"),
                    "turns": ev.get("num_turns"), "tool_calls": ev.get("tool_calls_count"),
                    "input_tokens": tu.get("input_tokens"), "output_tokens": tu.get("output_tokens"),
                    "reasoning_tokens": tu.get("reasoning_tokens"), "total_tokens": tu.get("total_tokens"),
                    "latency_ms": pf.get("latency_ms"), "error": r.get("error"),
                })
        progress("  %s: %d/%d files, %d attempts" % (col, i + 1, len(files), len(rows)), i + 1, len(files))
    if files and sys.stdout.isatty():
        print("")
    return cover, rows


def write_attempts(col, rows):
    out = DATA.parent / ("attempts_%s.csv" % col)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ATTEMPT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("  wrote %s (%d rows)" % (out, len(rows)))


# ---------------------------------------------------------------- reporting

def show(col, cover, label):
    tasks = set(t for m in cover for t in cover[m])
    common = set.intersection(*(set(cover[m]) for m in cover)) if cover else set()
    print("\n## %s from %s: %d tasks, %d models, %d tasks covered by all"
          % (col, label, len(tasks), len(cover), len(common)))
    print("  %-40s %6s %9s %12s" % ("model", "tasks", "attempts", "min/med/max"))
    for m in sorted(cover, key=lambda m: -sum(cover[m].values())):
        v = sorted(cover[m].values())
        print("  %-40s %6d %9d %12s" % (m, len(v), sum(v), "%d/%d/%d" % (v[0], v[len(v) // 2], v[-1])))


def diff(col, agg, smp):
    print("\n## %s: aggregates vs samples" % col)
    models = sorted(set(agg) | set(smp))
    total_mismatch = 0
    for m in models:
        a, s = agg.get(m, {}), smp.get(m, {})
        only_a, only_s = set(a) - set(s), set(s) - set(a)
        mism = [(t, a[t], s[t]) for t in set(a) & set(s) if a[t] != s[t]]
        total_mismatch += len(mism) + len(only_a) + len(only_s)
        print("  %-40s tasks only-in-agg=%d only-in-samples=%d attempt-count-mismatch=%d/%d"
              % (m, len(only_a), len(only_s), len(mism), len(set(a) & set(s))))
        for t, x, y in sorted(mism, key=lambda z: -abs(z[1] - z[2]))[:3]:
            print("      %-40s agg=%d samples=%d" % (t[:40], x, y))
    verdict = "MATCH" if total_mismatch == 0 else "%d discrepancies" % total_mismatch
    print("  => %s" % verdict)
    return total_mismatch


def consistency(col, rows):
    """Repeat-attempt disagreement and cost spread, pooled and within-shard.

    Pooled mixes attempts from different shards, i.e. different strategies and
    token budgets -- the study's own interventions. Within-shard holds the config
    fixed, so what remains is closer to stochastic variance. Report both; only the
    within-shard number is a claim about the agent rather than about the study.
    """
    def report(label, keyfn):
        per = collections.defaultdict(list)
        for r in rows:
            per[keyfn(r)].append(r)
        multi = {k: v for k, v in per.items() if len(v) >= 2}
        flips = sum(1 for v in multi.values() if len(set(passed(r["score"]) for r in v)) > 1)
        ratios = []
        for v in multi.values():
            toks = [r["total_tokens"] for r in v if passed(r["score"]) and r["total_tokens"]]
            if len(toks) >= 2:
                ratios.append(max(toks) / float(min(toks)))
        print("  %-28s groups>=2: %5d   disagree on pass/fail: %4d (%3.0f%%)"
              % (label, len(multi), flips, 100.0 * flips / len(multi) if multi else 0))
        if ratios:
            rs = sorted(ratios)
            print("  %-28s passing-attempt token spread: median %.1fx  p90 %.1fx  max %.1fx  (n=%d)"
                  % ("", statistics.median(rs), rs[int(0.9 * (len(rs) - 1))], rs[-1], len(rs)))

    print("\n## %s: consistency across repeated attempts" % col)
    report("pooled (model, task)", lambda r: (r["model"], r["task"]))
    report("within (model, task, shard)", lambda r: (r["model"], r["task"], r["shard"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["aggregates", "samples", "both"], default="both")
    ap.add_argument("--collection", choices=COLLECTIONS)
    args = ap.parse_args()
    cols = [args.collection] if args.collection else COLLECTIONS
    for col in cols:
        if not (DATA / "data" / col).exists():
            print("\n## %s: nothing pulled yet (python -m tally.pull)" % col)
            continue
        agg = smp = None
        if args.source in ("aggregates", "both"):
            agg, strat, unparsed = from_aggregates(col)
            show(col, agg, "aggregates")
            print("  strategies: %s   unparsed shards (assumed 1 epoch): %d" % (dict(strat), unparsed))
        if args.source in ("samples", "both"):
            smp, rows = from_samples(col)
            if not rows:
                print("\n## %s: no sample files yet (python -m tally.pull --phase samples)" % col)
                continue
            show(col, smp, "samples")
            write_attempts(col, rows)
            consistency(col, rows)
        if agg is not None and smp:
            diff(col, agg, smp)


if __name__ == "__main__":
    main()
