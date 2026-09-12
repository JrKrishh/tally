"""Step 4, cold start: can a reasoning model estimate task difficulty from the text?

Step 3's savings need a per-task difficulty. On a benchmark with history it comes
from other models' pass rates (0.87-0.94 AUC). On a benchmark nobody has run there
is no history, so the prior has to be read off the task itself. That is the one job
in this harness a reasoning model is genuinely needed for -- and it has a measured
ceiling to be judged against, not a vibe.

  python -m tally.coldstart extract                 # task text + true difficulty, no key needed
  python -m tally.coldstart score [--model ID]      # Nemotron on Token Factory, resumable, prints the receipt

Kill criterion: Spearman >= 0.3 against true difficulty AND beats the text-length
baseline, or it is decoration and the README says so.
"""
import argparse
import json
import random
import re
import sys

import numpy as np

from . import nebius, select
from .pull import COLLECTIONS, DATA, progress

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INSTR_CAP = 8000

PROMPT = """You are estimating how hard a task is for a strong frontier coding agent (GPT-5-class or Claude-Opus-class) working autonomously in a terminal with bash and python, a generous token budget, and several submission attempts.

TASK
----
{instruction}
----

Estimate the probability, from 0.0 to 1.0, that such an agent solves this task correctly. Weigh: how precisely the requirements are specified, how many independent things must all go right, whether the agent can verify its own work from inside the environment, and how much specialised knowledge it needs.

Think as much as you need. Then your final answer must be ONLY a JSON object on one line: {{"p_solve": <number between 0 and 1>, "why": "<under 15 words>"}}"""


# ---------------------------------------------------------------- extract

def cmd_extract(args):
    for col in COLLECTIONS:
        by, models, common = select.cells(select.load_attempts(col))
        tasks = sorted(set(t for _, t in by))
        diff = {}
        for t in tasks:
            rates = [np.mean([p for p, _ in by[(m, t)]]) for m in models if (m, t) in by]
            diff[t] = (float(np.mean(rates)), len(rates))
        text = {}
        files = sorted((DATA / "data" / col).rglob("*_samples.jsonl"))
        for i, p in enumerate(files):
            with p.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    t = rec.get("sample_id")
                    if t in text or t not in diff:
                        continue
                    text[t] = (rec.get("input") or {}).get("raw") or ""
            progress("  %s: %d/%d files, %d/%d tasks with text" % (col, i + 1, len(files), len(text), len(tasks)), i + 1, len(files))
            if len(text) == len(tasks):
                break
        out = DATA.parent / ("tasks_%s.jsonl" % col)
        with out.open("w", encoding="utf-8") as fh:
            for t in tasks:
                raw = text.get(t, "")
                fh.write(json.dumps({"task": t, "difficulty": diff[t][0], "n_models": diff[t][1],
                                     "chars": len(raw), "instruction": raw[:INSTR_CAP]}) + "\n")
        missing = sum(1 for t in tasks if not text.get(t))
        print("\n  wrote %s: %d tasks, %d without text" % (out, len(tasks), missing))


# ---------------------------------------------------------------- score

def parse_p(text, strict=False):
    """Last explicit "p_solve": x wins (a reasoning trace may revise itself). Only the
    final answer field may fall back to a bare number; reasoning text is full of them."""
    ms = re.findall(r'"?p_solve"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
    if ms:
        return max(0.0, min(1.0, float(ms[-1])))
    if strict:
        return None
    m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    return max(0.0, min(1.0, float(m.group(1)))) if m else None


def cmd_score(args):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model)
    for col in COLLECTIONS:
        src = DATA.parent / ("tasks_%s.jsonl" % col)
        if not src.exists():
            sys.exit("no %s -- run: python -m tally.coldstart extract" % src)
        tasks = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
        out = DATA.parent / ("coldstart_%s_%s.jsonl" % (col, slug))
        done = {}
        if out.exists():
            for l in out.open(encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    if r.get("p_solve") is not None:      # unparsed replies are retried, not skipped
                        done[r["task"]] = r
        usage_in = usage_out = usage_think = scored_now = 0
        with out.open("a", encoding="utf-8") as fh:
            for i, t in enumerate(tasks):
                if t["task"] in done or not t["instruction"]:
                    continue
                if args.limit and scored_now >= args.limit:
                    break
                scored_now += 1
                reply, reasoning, usage = nebius.chat(args.model, PROMPT.format(instruction=t["instruction"]))
                usage_in += usage.get("prompt_tokens", 0)
                usage_out += usage.get("completion_tokens", 0)
                usage_think += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
                p = parse_p(reply)
                if p is None and reasoning:
                    p = parse_p(reasoning, strict=True)
                r = {"task": t["task"], "p_solve": p, "reply": reply[:300],
                     "reasoning_chars": len(reasoning), "content_chars": len(reply)}
                if p is not None:
                    done[t["task"]] = r
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                progress("  %s: %d/%d scored  (%d in / %d out tokens this run)"
                         % (col, len(done), len(tasks), usage_in, usage_out), i + 1, len(tasks))
        report(col, tasks, done, args.model, usage_in, usage_out)


def cmd_report(args):
    """Re-run the analysis on scores already on disk. Spends nothing."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model)
    for col in COLLECTIONS:
        src = DATA.parent / ("tasks_%s.jsonl" % col)
        out = DATA.parent / ("coldstart_%s_%s.jsonl" % (col, slug))
        if not (src.exists() and out.exists()):
            print("## %s: nothing scored yet" % col)
            continue
        tasks = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
        done = {}
        for l in out.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("p_solve") is not None:
                    done[r["task"]] = r
        report(col, tasks, done, args.model, 0, 0)


def report(col, tasks, done, model, usage_in, usage_out):
    from scipy.stats import spearmanr
    rows = [(t["difficulty"], done[t["task"]]["p_solve"], t["chars"]) for t in tasks
            if t["task"] in done and done[t["task"]]["p_solve"] is not None]
    if len(rows) < 5:
        print("\n## %s: only %d scored" % (col, len(rows)))
        return
    true = [r[0] for r in rows]
    pred = [r[1] for r in rows]
    chars = [r[2] for r in rows]
    sp_model = spearmanr(true, pred).correlation
    sp_len = spearmanr(true, chars).correlation
    receipt = ("receipt this run: %d in / %d out tokens" % (usage_in, usage_out)) if usage_in else "report only"
    print("\n## %s cold start: %s on %d of %d tasks   (%s)" % (col, model, len(rows), len(tasks), receipt))
    rc = [done[t["task"]].get("reasoning_chars") for t in tasks if t["task"] in done]
    cc = [done[t["task"]].get("content_chars") for t in tasks if t["task"] in done]
    if any(x is not None for x in rc):
        print("  reply shape: mean %.0f chars reasoning field, %.0f chars content" %
              (np.mean([x or 0 for x in rc]), np.mean([x or 0 for x in cc])))
    print("  Spearman vs true difficulty:   model %.3f     text-length baseline %.3f" % (sp_model, sp_len))
    print("  kill criterion (>= 0.3 and beats length): %s"
          % ("PASS" if sp_model >= 0.3 and sp_model > abs(sp_len) else "FAIL"))
    # the downstream test: run step 3's policy with the model's difficulty instead of history.
    # A task the model could not score is unknown, and unknown means run it: prior 0.5.
    by, models, common = select.cells(select.load_attempts(col))
    p = dict((t["task"], done[t["task"]]["p_solve"]) for t in tasks
             if t["task"] in done and done[t["task"]]["p_solve"] is not None)
    missing = [t for t in common if t not in p]
    if missing:
        print("  (%d shared tasks unscored -> treated as uncertain, i.e. run)" % len(missing))
        for t in missing:
            p[t] = 0.5
    acc, cost = select.full(by, models, common)
    total = sum(cost.values())
    print("  step-3 policy [0.10, 0.90] cap 3, prior from:      cost   spearman   acc MAE")
    # third row is the honest baseline: no prior at all, cap 3, run every task
    for label, fn, lo, hi in (("history (other models)", None, 0.10, 0.90),
                              ("%s, no history" % model.split("/")[-1], lambda m, t: p[t], 0.10, 0.90),
                              ("no prior (run every task, cap 3)", None, 0.0, 1.0)):
        sp, mae, cf = [], [], []
        for seed in range(select.SEEDS):
            est, spent, _ = select.simulate(by, models, common, lo, hi, 3, random.Random(seed), diff_fn=fn)
            sp.append(spearmanr([acc[m] for m in models], [est[m] for m in models]).correlation)
            mae.append(float(np.mean([abs(est[m] - acc[m]) for m in models])))
            cf.append(sum(spent.values()) / total)
        print("    %-44s %5.1f%%   %8.3f   %7.3f" % (label, 100 * np.mean(cf), np.mean(sp), np.mean(mae)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("extract")
    s = sub.add_parser("score")
    s.add_argument("--model", default=nebius.DEFAULT_MODEL)
    s.add_argument("--limit", type=int, help="score at most this many new tasks per collection (smoke test)")
    r = sub.add_parser("report")
    r.add_argument("--model", default=nebius.DEFAULT_MODEL)
    args = ap.parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "score":
        cmd_score(args)
    elif args.cmd == "report":
        cmd_report(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
