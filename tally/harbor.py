"""Step 7: does the plan hold beyond two benchmarks and six models?

History from Harbor-Adapter (huggingface.co/datasets/kendx/Harbor-Adapter): Harbor trials
of 16 models under 6 agents on 60 benchmarks, up to 5 trials per (task, model, agent),
released with arXiv 2609.04298. Reward and tokens live inside each trial's archive, so
`pull` downloads one shard at a time, keeps one row per trial and deletes the shard.
`replay` then runs step 3's plans on each benchmark with select.simulate unchanged. A
system is a (model, agent) pair; its prior comes only from systems of other models, so
the held-out model is new to history, not just its agent.

  python -m tally.harbor pull --benchmark bfcl [--scratch DIR]
  python -m tally.harbor replay [--benchmark bfcl] [--seeds 200]
"""
import argparse
import collections
import csv
import io
import json
import random
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import requests

from . import select
from .pull import DATA

REPO = "kendx/Harbor-Adapter"
OUT = DATA.parent / "harbor"
TREE = "https://huggingface.co/api/datasets/%s/tree/main/data/harbor_adapters/%s"
FIELDS = ["benchmark", "task", "model", "agent", "trial_id", "reward", "n_input_tokens",
          "n_cache_tokens", "n_output_tokens", "cost_usd", "exception"]
# (label, window lo, window hi, attempts per uncertain task, attempts per certain task)
PLANS = [("default: certain once, uncertain twice", 0.10, 0.90, 2, 1),
         ("skip certain, 3 attempts (old default)", 0.10, 0.90, 3, 0),
         ("no history: every task twice", 0.0, 1.0, 2, 0),
         ("no history: every task once", 0.0, 1.0, 1, 0)]
MIN_COVER = 0.9                # a system joins if it has trials on this share of the benchmark's tasks


def manifest():
    """trial_id -> (benchmark, task, model, agent)."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download(REPO, "harbor_adapters.manifest.parquet", repo_type="dataset", local_dir=str(OUT))
    cells = {}
    for row in pq.read_table(path).to_pylist():
        for tid in row["trial_ids"]:
            cells[tid] = (row["benchmark"], row["task_name"], row["model"], row["agent"])
    return cells


def trial(archive):
    """Reward, tokens and exception from one trial archive; stops reading at the verifier."""
    result, reward = None, None
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r|gz") as tar:
        for member in tar:
            name = member.name.split("/", 1)[-1]
            if name == "result.json":
                result = json.load(tar.extractfile(member))
            elif name == "verifier/reward.txt":
                reward = tar.extractfile(member).read().decode("utf-8", "replace").strip()
            if result is not None and reward is not None:
                break
    ar = (result or {}).get("agent_result") or {}
    rewards = ((result or {}).get("verifier_result") or {}).get("rewards") or {}
    value = rewards.get("reward", reward)
    return {"reward": "" if value in (None, "") else float(value),
            "n_input_tokens": ar.get("n_input_tokens") or 0, "n_cache_tokens": ar.get("n_cache_tokens") or 0,
            "n_output_tokens": ar.get("n_output_tokens") or 0, "cost_usd": ar.get("cost_usd") or "",
            "exception": ((result or {}).get("exception_info") or {}).get("exception_type") or ""}


def pull(benchmark, scratch, cells):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    OUT.mkdir(parents=True, exist_ok=True)
    shards = sorted(e["path"] for e in requests.get(TREE % (REPO, benchmark), timeout=120).json() if e.get("type") == "file")
    out = OUT / ("trials_%s.csv" % benchmark)
    done = OUT / ("trials_%s.done" % benchmark)
    finished = set(done.read_text().split()) if done.exists() else set()
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for shard in shards:
            if shard in finished:
                continue
            for attempt in range(30):          # home connections drop; hf_hub_download resumes the partial file
                try:
                    local = Path(hf_hub_download(REPO, shard, repo_type="dataset", local_dir=str(scratch)))
                    break
                except Exception as e:
                    print("  %s: %s, retrying in %ds" % (shard, type(e).__name__, 10 * (attempt + 1)), flush=True)
                    time.sleep(10 * (attempt + 1))
            else:
                sys.exit("gave up on %s" % shard)
            rows = []                          # written only once the whole shard has been read, so a crash never half-writes it
            for batch in pq.ParquetFile(local).iter_batches(columns=["trial_id", "archive"], batch_size=32):
                for tid, archive in zip(batch.column("trial_id").to_pylist(), batch.column("archive").to_pylist()):
                    bench, task, model, agent = cells.get(tid, (benchmark, "", "", ""))
                    try:
                        fields = trial(archive)
                    except (tarfile.TarError, OSError, ValueError, EOFError) as e:
                        fields = {"reward": "", "n_input_tokens": 0, "n_cache_tokens": 0, "n_output_tokens": 0,
                                  "cost_usd": "", "exception": "unreadable archive: %s" % type(e).__name__}
                    rows.append(dict(benchmark=bench, task=task, model=model, agent=agent, trial_id=tid, **fields))
            w.writerows(rows)
            f.flush()
            n = len(rows)
            local.unlink()                     # keep the rows, not the 0.1-1 GB shard
            with done.open("a") as d:
                d.write(shard + "\n")
            print("  %s: %d trials" % (shard, n), flush=True)
    print("wrote %s" % out)


def history(benchmark):
    """by[(system, task)] = [(passed, tokens)] over the systems that cover the benchmark, plus
    each system's prior per task taken only from systems of other models."""
    rows = list(csv.DictReader((OUT / ("trials_%s.csv" % benchmark)).open(encoding="utf-8")))
    by = collections.defaultdict(list)
    fractional = 0
    for r in rows:
        if r["reward"] == "" or not r["task"]:
            continue                           # no verdict: the trial failed before the verifier ran
        score = float(r["reward"])
        fractional += 0.0 < score < 1.0
        tokens = float(r["n_input_tokens"]) + float(r["n_output_tokens"])
        by[(r["model"] + " | " + r["agent"], r["task"])].append((1.0 if score >= 1.0 else 0.0, tokens))
    tasks = sorted(set(t for _, t in by))
    systems = sorted(s for s in set(s for s, _ in by)
                     if sum((s, t) in by for t in tasks) >= MIN_COVER * len(tasks))
    common = [t for t in tasks if all((s, t) in by for s in systems)]
    model = dict((s, s.split(" | ")[0]) for s in systems)
    rate = dict(((s, t), float(np.mean([p for p, _ in by[(s, t)]]))) for s in systems for t in common)
    prior = {}
    for s in systems:
        others = [o for o in systems if model[o] != model[s]]
        prior[s] = dict((t, float(np.mean([rate[(o, t)] for o in others]))) for t in common)
    return by, systems, common, prior, fractional / max(1, len(rows))


def replay(benchmark, seeds):
    by, systems, common, prior, fractional = history(benchmark)
    acc, cost = select.full(by, systems, common)
    total = sum(cost.values())
    trials = sum(len(by[(s, t)]) for s in systems for t in common)
    out = {"benchmark": benchmark, "systems": len(systems), "models": len(set(s.split(" | ")[0] for s in systems)),
           "tasks": len(common), "trials": trials, "fractional": round(fractional, 3), "plans": {}}
    if len(systems) < 8 or len(common) < 30:
        return out
    top = max(systems, key=lambda s: acc[s])
    for label, lo, hi, k, kc in PLANS:
        cf, sp, mae, kept = [], [], [], []
        for seed in range(seeds):
            est, spent, _ = select.simulate(by, systems, common, lo, hi, k, random.Random(seed),
                                            diff_fn=lambda s, t: prior[s][t], k_certain=kc)
            cf.append(sum(spent.values()) / total)
            sp.append(select.spearman([acc[s] for s in systems], [est[s] for s in systems]))
            mae.append(float(np.mean([abs(est[s] - acc[s]) for s in systems])))
            kept.append(max(systems, key=lambda s: est[s]) == top)
        out["plans"][label] = {"cost": round(float(np.mean(cf)), 4), "spearman": round(float(np.mean(sp)), 3),
                               "mae": round(float(np.mean(mae)), 3), "top1": round(float(np.mean(kept)), 3)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--benchmark", required=True, action="append")
    p.add_argument("--scratch", default=str(OUT / "_shards"), help="where shards sit while they are read")
    r = sub.add_parser("replay")
    r.add_argument("--benchmark", action="append", help="default: every benchmark pulled")
    r.add_argument("--seeds", type=int, default=select.SEEDS)
    args = ap.parse_args()
    if args.cmd == "pull":
        OUT.mkdir(parents=True, exist_ok=True)
        cells = manifest()
        for b in args.benchmark:
            pull(b, Path(args.scratch), cells)
        return
    benches = args.benchmark or [p.stem[len("trials_"):] for p in sorted(OUT.glob("trials_*.csv")) if complete(p.stem[len("trials_"):])]
    results = []
    for b in benches:
        res = replay(b, args.seeds)
        results.append(res)
        print("\n## %s: %d systems (%d models), %d shared tasks, %d trials, %.1f%% fractional rewards"
              % (b, res["systems"], res["models"], res["tasks"], res["trials"], 100 * res["fractional"]))
        for label, v in res["plans"].items():
            print("  %-40s tokens %5.1f%%  spearman %.3f  MAE %.3f  top-1 %3.0f%%"
                  % (label, 100 * v["cost"], v["spearman"], v["mae"], 100 * v["top1"]))
    (OUT / "replay.json").write_text(json.dumps(results, indent=1))
    done = [r for r in results if r["plans"]]
    print("\n## across %d benchmarks (median; min-max)" % len(done))
    for label, *_ in PLANS:
        col = lambda key: [r["plans"][label][key] for r in done]
        print("  %-40s tokens %5.1f%% (%.1f-%.1f)  spearman %.3f (%.3f-%.3f)  MAE %.3f"
              % (label, 100 * np.median(col("cost")), 100 * min(col("cost")), 100 * max(col("cost")),
                 np.median(col("spearman")), min(col("spearman")), max(col("spearman")), np.median(col("mae"))))
    d, two, one = PLANS[0][0], PLANS[2][0], PLANS[3][0]
    print("  default at spearman >= 0.95: %d of %d" % (sum(r["plans"][d]["spearman"] >= 0.95 for r in done), len(done)))
    print("  default cheaper than every task twice: %d of %d; within 0.01 of its spearman: %d"
          % (sum(r["plans"][d]["cost"] < r["plans"][two]["cost"] for r in done), len(done),
             sum(r["plans"][d]["spearman"] >= r["plans"][two]["spearman"] - 0.01 for r in done)))
    print("  default ranks better than every task once: %d of %d"
          % (sum(r["plans"][d]["spearman"] > r["plans"][one]["spearman"] for r in done), len(done)))


def complete(benchmark):
    """Every shard of the benchmark has been read (a pull still running leaves a partial file)."""
    done = OUT / ("trials_%s.done" % benchmark)
    try:
        shards = [e for e in requests.get(TREE % (REPO, benchmark), timeout=60).json() if e.get("type") == "file"]
    except (requests.RequestException, ValueError):
        return False
    return done.exists() and len(done.read_text().split()) == len(shards)


if __name__ == "__main__":
    main()
