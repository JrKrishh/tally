"""tally report: score a real run against its plan, with the receipt.

Reads Harbor's job results for the run and validate phases and produces what
an evaluation is for: the new model's estimated accuracy on the whole
benchmark (measured where it ran, imputed from history where it didn't), its
rank among the models history knows, the tokens it spent against what the
full evaluation would have cost, and -- the honest part -- whether the cells
we chose NOT to run came out the way history said they would.

  python -m tally.report --plan data/plan_terminalbench.json [--jobs-dir jobs] [--model ...]
  python -m tally.report --plan data/plan_terminalbench.json --replay   # step 3's plans on this run's attempts
"""
import argparse
import json
import random
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


def model_price(model):
    """(usd per prompt token, usd per completion token) from Token Factory's models list, or None.
    The list only carries prices with ?verbose=true; no cached-token price is published."""
    try:
        import requests
        from . import nebius
        r = requests.get(nebius.API + "/models?verbose=true",
                         headers={"Authorization": "Bearer " + nebius.key()}, timeout=30)
        for m in r.json().get("data", []):
            if m.get("id") == model and m.get("pricing"):
                return float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])
    except (Exception, SystemExit):
        return None
    return None


def find_jobs(jobs_dir, prefix, include_dry=False):
    """Job dirs matching a prefix, oldest first. Names are timestamped by tally.run."""
    dirs = [p for p in Path(jobs_dir).glob(prefix + "*") if p.is_dir()]
    if not include_dry:
        dirs = [p for p in dirs if not p.name.endswith("-dry")]
    return sorted(dirs, key=lambda p: p.name)          # by the name's timestamp: a pull resets mtimes


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


REPLAY_PLANS = [  # label, attempts per uncertain task, attempts per certain task (0 imputes), uses history
    ("skip certain tasks, 3 attempts (the old default)", 3, 0, True),
    ("certain tasks once, uncertain twice (the default)", 2, 1, True),
    ("certain tasks once, uncertain 3 times", 3, 1, True),
    ("no history: every task twice", 2, 2, False),
    ("no history: every task once", 1, 1, False),
]


def replay_plans(run_dir, benchmark, lo=0.10, hi=0.90):
    """Step 3's plans replayed on a real run's own attempts: does a plan hold for a model the
    history never saw? Truth is the run's pass rate over the tasks it attempted in full; each
    plan subsamples those attempts, with certainty taken from history, over select.SEEDS samples.
    Tokens come from site.trial_rows, the per-trial counts the receipt uses."""
    from . import plan as planner, site
    atts = {}
    for r in site.trial_rows(run_dir):
        atts.setdefault(r["task"], []).append((1.0 if r["outcome"] == "pass" else 0.0, float(r["tokens"])))
    n = max(len(a) for a in atts.values())
    tasks = sorted(t for t, a in atts.items() if len(a) == n)
    diff = planner.history(benchmark)[0]
    certain = set(t for t in tasks if t in diff and not lo <= diff[t] <= hi)
    truth = float(np.mean([np.mean([p for p, _ in atts[t]]) for t in tasks]))
    full = sum(tok for t in tasks for _, tok in atts[t])
    print("## plans replayed on this run: %d tasks x %d attempts, pass rate %.3f; history calls %d of them certain"
          % (len(tasks), n, truth, len(certain)))
    print("  %-50s %7s %9s %7s %11s" % ("plan", "tokens", "estimate", "bias", "mean |err|"))
    for label, k, k_certain, use_history in REPLAY_PLANS:
        est, cost = [], []
        for seed in range(select.SEEDS):
            rng, total, spent = random.Random(seed), 0.0, 0.0
            for t in tasks:
                cap = k_certain if use_history and t in certain else k
                if not cap:
                    total += diff[t]                   # impute from history, spend nothing
                    continue
                sample = list(atts[t])
                rng.shuffle(sample)
                total += float(np.mean([p for p, _ in sample[:cap]]))
                spent += sum(tok for _, tok in sample[:cap])
            est.append(total / len(tasks))
            cost.append(spent / full)
        err = float(np.mean(np.abs(np.array(est) - truth)))
        print("  %-50s %6.0f%% %9.3f %+7.3f %11.3f" % (label, 100 * np.mean(cost), np.mean(est), np.mean(est) - truth, err))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--jobs-dir", default=str(ROOT / "jobs"))
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
    ap.add_argument("--inspect", action="store_true", help="dump the structure of the first trial result and exit")
    ap.add_argument("--replay", action="store_true", help="replay step 3's plans on the run's own attempts and exit")
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
    if args.replay:
        if not rdir:
            sys.exit("no run job to replay under %s" % args.jobs_dir)
        replay_plans(rdir, plan["benchmark"])
        return
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
    # An attempt is a trial that reached a verdict (a timeout scores 0 and counts). A trial
    # that never reached the task -- a 402, a dead sandbox -- is an infrastructure error.
    n_attempts = sum(1 for a in run.values() for x, _, _ in a if x is not None)
    errors = sum(1 for a in list(run.values()) + list(val.values()) for x, _, e in a if e and x is None)

    # estimate over the whole benchmark: measured where run, prior where skipped
    prior = plan["skipped_prior"]
    run_prior = plan.get("run_prior", {})            # plans written before this field lack it
    all_tasks = sorted(set(plan["tasks_run"]) | set(prior) | set(plan.get("tasks_no_history", [])))
    est_parts, covered, imputed_run = [], 0, 0
    for t in all_tasks:
        if t in measured:
            est_parts.append(measured[t]); covered += 1
        elif t in prior:
            est_parts.append(prior[t])
        elif t in run_prior:                          # planned but not yet run: history stands in for now
            est_parts.append(run_prior[t]); imputed_run += 1
    est = float(np.mean(est_parts)) if est_parts else float("nan")

    print("## %s on %s (%s)" % (args.model, plan["benchmark"], plan["dataset"]))
    print("  run phase   : %d tasks measured, %d attempts with a verdict, %d infrastructure errors" % (len(measured), n_attempts, errors))
    if not measured:
        print("  nothing measured: every trial errored, so there is no estimate and no ranking yet.")
        for t, atts in list(run.items())[:3]:
            print("   %s: %d attempt(s), errored=%s" % (t, len(atts), [e for _, _, e in atts]))
        return
    planned = set(plan["tasks_run"]) | set(plan.get("tasks_no_history", []))
    run_measured = [t for t in measured if t in planned]
    print("  measured pass rate on run tasks: %.3f   (%d of %d planned cells have a verdict)"
          % (float(np.mean([measured[t] for t in run_measured])) if run_measured else float("nan"), len(run_measured), len(planned)))
    print("  estimated accuracy on all %d tasks (measured where run, history where skipped): %.3f%s"
          % (len(all_tasks), est, "   <- mostly prior until more cells run" if len(run_measured) < len(planned) // 2 else ""))
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
        price = model_price(args.model)
        if price and rstats.get("n_input_tokens") is not None:
            usd = rstats["n_input_tokens"] * price[0] + (rstats.get("n_output_tokens") or 0) * price[1]
            print("  cost at list price: $%.2f   ($%.2f/M in, $%.2f/M out; no cached-token price is published, so cache is billed as input here)"
                  % (usd, price[0] * 1e6, price[1] * 1e6))
    else:
        print("  tokens: none recorded yet (job stats empty and no token counters in agent results)")

    # rank like for like: history's attempts without correctness feedback, on the same tasks
    acc, ref_tasks, excluded = select.nofeedback_reference(plan["benchmark"])
    new_parts = []
    for t in ref_tasks:
        if t in measured:
            new_parts.append(measured[t])
        elif t in prior:
            new_parts.append(prior[t])
        elif t in run_prior:
            new_parts.append(run_prior[t])
    est_ref = float(np.mean(new_parts)) if new_parts else float("nan")
    table = sorted([(a, m) for m, a in acc.items()] + [(est_ref, args.model + "  <- new, estimated")], reverse=True)
    print("\n  ranking on the %d tasks shared by history models' runs without correctness feedback:" % len(ref_tasks))
    for i, (a, m) in enumerate(table, 1):
        print("   %d. %-50s %.3f" % (i, m, a))
    if excluded:
        print("   not ranked, too few runs without feedback: %s"
              % ", ".join("%s (%d tasks)" % (m, n) for m, n in sorted(excluded.items())))

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
