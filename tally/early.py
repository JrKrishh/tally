"""Step 2 kill check: does a trajectory's prefix predict its outcome?

Early stopping only works if something visible in the first N agent turns
predicts final pass/fail. Before any reasoning model is asked to read
trajectories, ask whether trivial features already do -- tokens so far, tool
calls, error-looking results, repeated commands. If a logistic regression on
those scores ~0.5 AUC on held-out TASKS, nothing in the prefix carries signal
and early stopping is dead regardless of the model.

  python -m tally.early extract      # one pass over the samples -> data/prefix_<col>.csv
  python -m tally.early fit          # grouped CV by task, AUC per cutoff N

Held-out unit is the task, not the attempt: a predictor that has seen other
attempts at the same task is just memorising task difficulty.
"""
import argparse
import csv
import json
import re
import sys

import numpy as np

from .pull import COLLECTIONS, DATA, progress
from .matrix import passed, shard_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CUTOFFS = [3, 5, 10, 20, 40]
TEXT_N, TEXT_CAP = 10, 6000      # raw prefix text kept at this cutoff, last TEXT_CAP chars
ERR = re.compile(r"traceback|error|no such file|command not found|failed|exception|permission denied", re.I)
FEATS = ["msgs", "tool_calls", "bash", "python", "submit", "err_results", "repeat_calls",
         "chars_assistant", "chars_tool_out", "ended_before_n"]


# ---------------------------------------------------------------- extract

def prefix_rows(rec, col, model, shard, strategy):
    """One row per cutoff N with cumulative features over the first N assistant turns,
    plus the raw prefix text at TEXT_N for the content baseline."""
    msgs = rec.get("messages") or []
    ev = rec.get("evaluation") or {}
    label = 1 if passed(ev.get("score")) else 0
    seen_calls, out = set(), []
    st = dict((f, 0) for f in FEATS)
    assistant_turns, next_cut = 0, 0
    total_assistant = sum(1 for m in msgs if m.get("role") == "assistant")
    text_parts, text_rec = [], None

    def snapshot(n, ended):
        row = {"collection": col, "model": model, "task": rec["sample_id"], "shard": shard,
               "strategy": strategy, "n": n, "label": label,
               "total_turns": total_assistant, "total_tokens": (rec.get("token_usage") or {}).get("total_tokens")}
        row.update(st)
        row["ended_before_n"] = int(ended)
        if n == TEXT_N:
            text_rec_ = {"model": model, "task": rec["sample_id"], "shard": shard, "label": label,
                         "ended_before_n": int(ended), "text": "".join(text_parts)[-TEXT_CAP:]}
            row["_text"] = text_rec_
        return row

    for m in msgs:
        role = m.get("role")
        st["msgs"] += 1
        if role == "assistant":
            assistant_turns += 1
            text_parts.append("\n[A] " + (m.get("content") or ""))
            for tc in m.get("tool_calls") or []:
                text_parts.append("\n[CALL %s] %s" % (tc.get("name"), json.dumps(tc.get("arguments"))[:600]))
            st["chars_assistant"] += len(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                name = (tc.get("name") or "").lower()
                st["tool_calls"] += 1
                if name == "bash":
                    st["bash"] += 1
                elif name == "python":
                    st["python"] += 1
                elif "submit" in name:
                    st["submit"] += 1
                key = (name, json.dumps(tc.get("arguments"), sort_keys=True))
                if key in seen_calls:
                    st["repeat_calls"] += 1
                seen_calls.add(key)
        elif role == "tool":
            c = m.get("content") or ""
            text_parts.append("\n[OUT] " + c[:1500])
            st["chars_tool_out"] += len(c)
            if ERR.search(c[:4000]):
                st["err_results"] += 1
        while next_cut < len(CUTOFFS) and assistant_turns >= CUTOFFS[next_cut]:
            out.append(snapshot(CUTOFFS[next_cut], False))
            next_cut += 1
    while next_cut < len(CUTOFFS):          # trajectory ended before this cutoff: prefix == whole run
        out.append(snapshot(CUTOFFS[next_cut], True))
        next_cut += 1
    return out


def cmd_extract(args):
    for col in COLLECTIONS:
        files = sorted((DATA / "data" / col).rglob("*_samples.jsonl"))
        if not files:
            print("## %s: no samples (python -m tally.pull)" % col)
            continue
        out = DATA.parent / ("prefix_%s.csv" % col)
        out_text = DATA.parent / ("prefix_text_%s.jsonl" % col)
        n_rows = 0
        with out.open("w", newline="", encoding="utf-8") as fh, out_text.open("w", encoding="utf-8") as ft:
            w = None
            for i, p in enumerate(files):
                model = p.parent.parent.name + "/" + p.parent.name
                shard = p.name[:-len("_samples.jsonl")]
                strategy, _ = shard_config(p.parent / (shard + ".json"))
                with p.open(encoding="utf-8", errors="replace") as src:
                    for line in src:
                        if not line.strip():
                            continue
                        for row in prefix_rows(json.loads(line), col, model, shard[:8], strategy):
                            text_rec = row.pop("_text", None)
                            if text_rec:
                                ft.write(json.dumps(text_rec) + "\n")
                            if w is None:
                                w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                                w.writeheader()
                            w.writerow(row)
                            n_rows += 1
                progress("  %s: %d/%d files, %d prefix rows" % (col, i + 1, len(files), n_rows), i + 1, len(files))
        print("\n  wrote %s (%d rows) and %s" % (out, n_rows, out_text.name))


# ---------------------------------------------------------------- fit

def load(col):
    p = DATA.parent / ("prefix_%s.csv" % col)
    if not p.exists():
        sys.exit("no %s -- run: python -m tally.early extract" % p)
    with p.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # Some Terminal-Bench runs were deposited twice under different shard uuids
    # (identical tokens, turns and latency to the millisecond). Keep the first copy.
    seen, out, dropped = set(), [], 0
    for r in rows:
        k = (r["model"], r["task"], r["n"], r["label"], r["total_turns"], r["total_tokens"])
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        out.append(r)
    if dropped:
        print("  %s: dropped %d prefix rows (%d trajectories) as cross-shard duplicates"
              % (col, dropped, dropped // len(CUTOFFS)))
    return out


def auc_grouped_cv(rows, feat_names, with_model, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    models = sorted(set(r["model"] for r in rows))
    X, y, g = [], [], []
    for r in rows:
        f = [float(r[k]) for k in feat_names]
        if with_model:
            f += [1.0 if r["model"] == m else 0.0 for m in models]
        X.append(f)
        y.append(int(r["label"]))
        g.append(r["task"])
    X, y = np.log1p(np.array(X)), np.array(y)
    if len(set(y)) < 2:
        return float("nan")
    aucs = []
    for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
        if len(set(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def auc_text(col, folds=5):
    """TF-IDF over the raw prefix text at TEXT_N, held out by task, still-running only.
    If this beats the count features, the content carries signal the counts miss and a
    reader of the trajectory has a job. If it doesn't, the prefix is uninformative."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    p = DATA.parent / ("prefix_text_%s.jsonl" % col)
    recs = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    recs = [r for r in recs if not r["ended_before_n"]]
    seen, uniq = set(), []                       # same cross-shard duplicates as load()
    for r in recs:
        k = (r["model"], r["task"], r["label"], r["text"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    recs = uniq
    texts = [r["text"] for r in recs]
    y = np.array([r["label"] for r in recs])
    g = [r["task"] for r in recs]
    aucs = []
    for tr, te in GroupKFold(n_splits=folds).split(texts, y, g):
        vec = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        clf = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr, y[tr])
        if len(set(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs)), len(recs)


def auc_known_benchmark(rows, group_by="shard", prior_by=("model", "task"), folds=5):
    """Deployment scenarios where history is legitimate prior knowledge.
    group_by="shard", prior_by=(model, task): a NEW RUN of a model already evaluated.
    group_by="model", prior_by=(task,): a NEW MODEL on a known benchmark -- the prior
    is the task's pass rate across the other models, i.e. its difficulty.
    Prior is computed from training folds only, then combined with prefix features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    models = sorted(set(r["model"] for r in rows))
    y = np.array([int(r["label"]) for r in rows])
    g = [r[group_by] for r in rows]
    folds = min(folds, len(set(g)))
    base = [[float(r[k]) for k in FEATS] + [1.0 if r["model"] == m else 0.0 for m in models] for r in rows]
    base = np.log1p(np.array(base))
    out = {}
    for name, use_prefix in (("prior only", False), ("prior + prefix", True)):
        aucs = []
        for tr, te in GroupKFold(n_splits=folds).split(base, y, g):
            s, n = {}, {}
            for i in tr:
                t = tuple(rows[i][k] for k in prior_by)
                s[t] = s.get(t, 0) + y[i]
                n[t] = n.get(t, 0) + 1
            def prior(i):
                t = tuple(rows[i][k] for k in prior_by)
                return (s.get(t, 0) + 1.0) / (n.get(t, 0) + 2.0)
            Ptr = np.array([[prior(i)] for i in tr]); Pte = np.array([[prior(i)] for i in te])
            Xtr = np.hstack([base[tr], Ptr]) if use_prefix else Ptr
            Xte = np.hstack([base[te], Pte]) if use_prefix else Pte
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, y[tr])
            if len(set(y[te])) > 1:
                aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
        out[name] = float(np.mean(aucs))
    return out


def cmd_fit(args):
    for col in COLLECTIONS:
        rows = load(col)
        base = np.mean([int(r["label"]) for r in rows if r["n"] == str(CUTOFFS[0])])
        print("\n## %s: AUC of pass/fail from the first N agent turns  (base pass rate %.2f, %d trajectories)"
              % (col, base, sum(1 for r in rows if r["n"] == str(CUTOFFS[0]))))
        print("  %4s %10s %12s %14s %14s   %s" % ("N", "ended<N", "length-only", "all-features", "+model id", "AUC 0.5 = no signal"))
        for n in CUTOFFS:
            sub = [r for r in rows if r["n"] == str(n)]
            ended = np.mean([int(r["ended_before_n"]) for r in sub])
            a_len = auc_grouped_cv(sub, ["msgs", "tool_calls", "chars_assistant", "chars_tool_out"], False)
            a_all = auc_grouped_cv(sub, FEATS, False)
            a_mod = auc_grouped_cv(sub, FEATS, True)
            print("  %4d %9.0f%% %12.3f %14.3f %14.3f" % (n, 100 * ended, a_len, a_all, a_mod))
        # the honest control: same features on trajectories that have NOT ended by N
        print("  -- restricted to trajectories still running at N (no peeking at finished runs):")
        for n in CUTOFFS:
            sub = [r for r in rows if r["n"] == str(n) and r["ended_before_n"] == "0"]
            if len(sub) < 50:
                print("  %4d  (only %d still running)" % (n, len(sub)))
                continue
            a_all = auc_grouped_cv(sub, FEATS, False)
            a_mod = auc_grouped_cv(sub, FEATS, True)
            print("  %4d %9s %12s %14.3f %14.3f   n=%d" % (n, "", "", a_all, a_mod, len(sub)))
        # does the CONTENT of the prefix carry signal the counts miss?
        a_txt, n_txt = auc_text(col)
        print("  -- prefix TEXT (tf-idf) at N=%d, held out by task, still running:  AUC %.3f   n=%d" % (TEXT_N, a_txt, n_txt))
        # deployment scenario: known benchmark, historical per-task pass rate is fair game
        sub = [r for r in rows if r["n"] == str(TEXT_N) and r["ended_before_n"] == "0"]
        kb = auc_known_benchmark(sub)
        print("  -- known benchmark, SAME model, new run (held out by shard):   prior only %.3f   prior + prefix %.3f"
              % (kb["prior only"], kb["prior + prefix"]))
        kb2 = auc_known_benchmark(sub, group_by="model", prior_by=("task",))
        print("  -- known benchmark, NEW model (held out by model, prior = task difficulty from other models):"
              "   prior only %.3f   prior + prefix %.3f" % (kb2["prior only"], kb2["prior + prefix"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("extract")
    sub.add_parser("fit")
    args = ap.parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "fit":
        cmd_fit(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
