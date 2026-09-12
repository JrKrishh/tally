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


def find_jobs(jobs_dir, prefix, include_dry=False):
    """Job dirs matching a prefix, oldest first. Names are timestamped by tally.run."""
    dirs = [p for p in Path(jobs_dir).glob(prefix + "*") if p.is_dir()]
    if not include_dry:
        dirs = [p for p in dirs if not p.name.endswith("-dry")]
    return sorted(dirs, key=lambda p: p.stat().st_mtime)


def load_job(jobs_dir, prefix):
    """Newest job matching prefix -> ({task: [(reward, tokens, errored)]}, job stats, job dir)."""
    dirs = find_jobs(jobs_dir, prefix)
    if not dirs:
        return None, {}, None
    jd = dirs[-1]
    trials, stats = [], {}
    top = jd / "result.json"
    if top.exists():
        j = json.load(top.open(encoding="utf-8"))
        stats = j.get("stats") or {}
        trials = j.get("trial_results") or []
    salvaged = 0
    if not trials:                                   # per-trial dirs: jobs/<job>/<task>__<id>/
        for d in sorted(p for p in jd.iterdir() if p.is_dir()):
            rec, f = None, d / "result.json"
            if f.exists() and f.stat().st_size > 0:
                try:
                    rec = json.load(f.open(encoding="utf-8"))
                except Exception:
                    rec = None
            if rec is None:
                # Harbor can crash at the final write (cp1252 on Windows) after the agent and
                # verifier both finished. The verdict and the token counts are still on disk.
                rew, traj = d / "verifier" / "reward.txt", d / "agent" / "trajectory.json"
                if not rew.exists():
                    continue
                try:
                    reward = float(rew.read_text(encoding="utf-8").strip() or 0)
                except ValueError:
                    reward = None
                rec = {"task_name": d.name.split("__")[0], "verifier_result": {"rewards": {"reward": reward}}, "agent_result": {}}
                if traj.exists():
                    try:
                        rec["agent_result"] = {"metrics": json.load(traj.open(encoding="utf-8", errors="replace")).get("final_metrics") or {}}
                    except Exception:
                        pass
                salvaged += 1
            trials.append(rec)
    stats["n_salvaged"] = salvaged
    out = {}
    for t in trials:
        task = t.get("task_name") or t.get("task_id") or "?"
        rewards = ((t.get("verifier_result") or {}).get("rewards") or {})
        reward = rewards.get("reward") if isinstance(rewards, dict) else None
        errored = bool(t.get("exception_info"))
        out.setdefault(task, []).append((float(reward) if reward is not None else None, find_tokens(t.get("agent_result") or {}), errored))
    return out, stats, jd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--jobs-dir", default=str(ROOT / "jobs"))
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
    ap.add_argument("--inspect", action="store_true", help="dump the structure of the first trial result and exit")
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-")
    lo, hi = plan["window"]

    if args.inspect:
        dirs = find_jobs(args.jobs_dir, "tally-%s-run-%s" % (plan["benchmark"], slug), include_dry=True)
        jd = dirs[-1] if dirs else Path(args.jobs_dir) / "none"
        files = sorted(jd.rglob("*.json")) if jd.exists() else []
        print("## %s: %d json files" % (jd, len(files)))
        for f in files[:12]:
            print("   %s  (%d bytes)" % (f.relative_to(jd), f.stat().st_size))
        def shape(v, depth=0):
            if isinstance(v, dict):
                return "{" + ", ".join("%s: %s" % (k, shape(x, depth + 1)) for k, x in list(v.items())[:14]) + "}" if depth < 3 else "{...}"
            if isinstance(v, list):
                return "[%d x %s]" % (len(v), shape(v[0], depth + 1) if v else "?")
            return type(v).__name__
        for f in files:
            if f.name == "result.json":
                j = json.load(f.open(encoding="utf-8"))
                print("\n## %s" % f.relative_to(jd))
                print("   " + shape(j))
                tr = (j.get("trial_results") or [j])[0]
                for key in ("agent_result", "verifier_result", "agent_execution", "exception_info"):
                    if key in tr:
                        print("   %s: %s" % (key, json.dumps(tr[key])[:700]))
                break
        return

    run, rstats, rdir = load_job(args.jobs_dir, "tally-%s-run-%s" % (plan["benchmark"], slug))
    val, vstats, vdir = load_job(args.jobs_dir, "tally-%s-validate-%s" % (plan["benchmark"], slug))
    run, val = run or {}, val or {}
    if not run and not val:
        sys.exit("no results under %s for %s -- run: python -m tally.run" % (args.jobs_dir, args.model))
    if rdir:
        print("  run job: %s   trials: %s completed, %s errored, %s pending%s"
              % (rdir.name, rstats.get("n_completed_trials"), rstats.get("n_errored_trials"), rstats.get("n_pending_trials"),
                 "   (%d salvaged from trial dirs: Harbor never wrote their summary)" % rstats["n_salvaged"] if rstats.get("n_salvaged") else ""))

    def rate(atts):
        r = [x for x, _, _ in atts if x is not None]
        return float(np.mean([1.0 if x >= 1.0 else 0.0 for x in r])) if r else None

    # measured cells: the run phase, plus validation cells -- a validated cell was run, so it
    # is a measurement (one attempt), not a prior any more
    measured = dict((t, rate(a)) for t, a in run.items() if rate(a) is not None)
    for t, a in val.items():
        if t not in measured and rate(a) is not None:
            measured[t] = rate(a)
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
    if not measured:
        print("  nothing measured: every trial errored, so there is no estimate and no ranking yet.")
        for t, atts in list(run.items())[:3]:
            print("   %s: %d attempt(s), errored=%s" % (t, len(atts), [e for _, _, e in atts]))
        return
    n_planned = len(plan["tasks_run"]) + len(plan.get("tasks_no_history", []))
    print("  measured pass rate on run tasks: %.3f   (%d of %d planned cells measured so far)"
          % (float(np.mean(list(measured.values()))), covered, n_planned))
    print("  estimated accuracy on all %d tasks (measured where run, history where skipped): %.3f%s"
          % (len(all_tasks), est, "   <- mostly prior until more cells run" if covered < n_planned // 2 else ""))
    # Harbor tallies tokens and cost at the job level; prefer that over hunting in agent results.
    job_tokens = (rstats.get("n_input_tokens") or 0) + (rstats.get("n_output_tokens") or 0)
    if job_tokens:
        tokens_run = job_tokens
    if tokens_run:
        full_attempts = len(plan["tasks_run"]) * plan["attempts"] + len(plan["tasks_validate"]) + len(plan.get("tasks_no_history", [])) * plan["attempts"]
        per = tokens_run / max(1, n_attempts)
        print("  tokens spent (run phase): %.2fM  (%s in / %s out, cache %s)   Harbor cost_usd: %s"
              % (tokens_run / 1e6, rstats.get("n_input_tokens"), rstats.get("n_output_tokens"), rstats.get("n_cache_tokens"), rstats.get("cost_usd")))
        print("  per attempt: %.0fK tokens   -> the full plan (%d attempts) at this rate: ~%.0fM tokens"
              % (per / 1e3, full_attempts, per * full_attempts / 1e6))
        print("  plan expected ~%.0fM at frontier-model trajectory lengths" % (plan["expected_tokens"]["plan"] / 1e6))
    else:
        print("  tokens: none recorded yet (job stats empty and no token counters in agent results)")

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
        side = {"fail": [0, 0], "pass": [0, 0]}          # side -> [agree, total]
        for t, atts in sorted(val.items()):
            r = rate(atts)
            p = prior.get(t)
            if r is None or p is None:
                continue
            predicted = "pass" if p > hi else "fail"
            actual = "pass" if r >= 0.5 else "fail"
            side[predicted][1] += 1
            side[predicted][0] += predicted == actual
            print("   %-34s history %.2f -> predicted %-4s   actual %-4s   %s" % (t[:34], p, predicted, actual, "ok" if predicted == actual else "MISS"))
        total_agree = side["fail"][0] + side["pass"][0]
        total = side["fail"][1] + side["pass"][1]
        print("  agreement: %d/%d   fail-side %d/%d   pass-side %d/%d" % (total_agree, total, side["fail"][0], side["fail"][1], side["pass"][0], side["pass"][1]))
        if side["pass"][1] and side["pass"][0] < side["pass"][1]:
            print("  history's 'certain pass' is certain for the frontier tier, not for this model:")
            print("  re-plan skipping only the fail side ->  python -m tally.plan --lo %.2f --hi 1.0" % lo)


if __name__ == "__main__":
    main()
