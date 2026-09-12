# Tally

**Cost-aware evaluation for coding agents. Same ranking, a fraction of the spend, with the receipt.**

Status: **steps 1–2 complete.** The data is validated and the first mechanism has
been tested. No cost saving is claimed yet. This README grows only as claims are earned.

## The problem

Evaluating agents has become the compute bottleneck. One pass over SWE-bench
Multimodal runs over $2,200. A statistically credible evaluation — eight reruns per
cell — takes the Holistic Agent Leaderboard from $40K to roughly $320K. τ-bench
scores drop from 60% to 25% once consistency is checked. Static benchmarks solved
this with subsampling (100–200× fewer items, same rankings); agent benchmarks manage
2–3.5×. The people closest to the problem say rigorous evaluation is now gated to
well-funded labs.

## The data

[Every Eval Ever](https://github.com/evaleval/every_eval_ever) hosts the per-attempt
logs of *How Inference Compute Shapes Frontier LLM Evaluation*
([arXiv 2606.17930](https://arxiv.org/abs/2606.17930)): six frontier models — Opus 4,
4.5, 4.6 and GPT-5, 5.2, 5.4 — on Terminal-Bench and SWE-bench Pro, five independent
trajectories per task per condition, every trajectory logged with outcome, turns,
tool calls, tokens and latency. Its authors ask people to reuse it instead of
re-running baselines. This project does.

## Step 1: does the data say what its metadata says?

No. The shard aggregates list `sample_ids` and encode epochs as `+Nep`, so the
model × task × attempts matrix looks reconstructible without touching the 7.4 GB of
samples. Rebuilding it both ways and diffing:

| | from aggregates | from samples (truth) |
|---|---|---|
| Terminal-Bench tasks / covered by all six models | 89 / 73 | **88 / 86** |
| Terminal-Bench attempts, median per (model, task) | 2–5 | **10–15** |
| SWE-bench Pro tasks / covered by all six | 54 / 40 | 54 / 40 |
| SWE-bench Pro attempts, median per (model, task) | 5–10 | **10–14** |
| total attempts | ~6,400 implied | **13,915** |

Every (model, task) pair mismatched, always in the same direction: **the aggregates
undercount by 2–3×.** Adaptive sampling means attempts per task vary within a shard,
so `+Nep` is a ceiling, and eight generically-named shards carry 1,600 SWE-bench Pro
attempts with no task list at all. Shard metadata is not a substitute for the
samples. `data/attempts_<collection>.csv` — one row per trajectory — is the ground
truth everything downstream builds on. 76 Terminal-Bench trajectories were deposited
twice under different shard ids (identical tokens, turns and latency to the
millisecond); they are dropped.

The median of ~10 attempts per pair is 5 trajectories × 2 conditions, which is the
paper's stated design. That is the second, independent confirmation.

### Decoding the two conditions

`S-adaptive` and `S-adaptive+C` are the paper's *no feedback* and *oracle
correctness feedback* arms. Not inferred from the name — read off the stop reasons:

| condition | ended on successful submit | ended on repeated answer |
|---|---|---|
| `S-adaptive` (no feedback) | **0%** | 67–92% |
| `S-adaptive+C` (oracle feedback) | 62–66% | 25–33% |

Without feedback a trajectory cannot know it has succeeded, so it can only end by
repeating itself or hitting the budget. The `repetition_guard` stop — 56% of all
trajectories — is therefore the *designed* termination of the no-feedback arm, not
agents pathologically looping. Worth stating because it is easy to misread.

### Consistency: what is variance and what is the experiment

Pooling attempts across shards mixes the two arms. Grouping within a shard holds the
condition fixed, so what remains is stochastic:

| | pooled | within one condition |
|---|---|---|
| Terminal-Bench pairs that disagree on pass/fail | 45% | **12%** |
| Terminal-Bench token spread among *passing* attempts, median / p90 | 31.7× / 310× | **2.3× / 53×** |
| SWE-bench Pro pairs that disagree | 34% | **12%** |
| SWE-bench Pro token spread among passing attempts | 16× / 675× | **2.1× / 11×** |

The pooled numbers are dramatic and mostly artefacts of the study design. The
within-condition numbers are the claim: **one attempt in eight flips, and two passing
attempts on the same task under the same config routinely differ 2× in cost, with a
long tail.** That is the variance any cheaper evaluation has to reproduce, not
average away.

## Step 2: does a trajectory's prefix predict its outcome?

The obvious mechanism for a cheaper evaluation is early stopping: read the first *N*
turns, predict failure, kill the run. Tested before any model was asked to do the
reading. All AUCs are on trajectories still running at turn *N* — a run that has
already finished has its outcome baked in — and held out by **task**, so nothing can
be memorised about the task itself.

| pass/fail from the first 10 turns, unseen tasks | count features | prefix text (TF-IDF) |
|---|---|---|
| Terminal-Bench | 0.52 | 0.63 |
| SWE-bench Pro | 0.64 | **0.84** |

Counts are near coin-flip. Content carries real signal on SWE-bench Pro — the kind of
thing a reasoning model would be asked to extract. Then the second experiment made
the question moot:

| pass/fail from the first 10 turns, benchmark seen before | history alone | history + prefix |
|---|---|---|
| Terminal-Bench, same model, new run | 0.883 | 0.884 |
| Terminal-Bench, **new model**, prior = task difficulty from the other five | **0.868** | 0.868 |
| SWE-bench Pro, same model, new run | 0.947 | 0.948 |
| SWE-bench Pro, **new model**, prior = task difficulty from the other five | **0.938** | 0.933 |

**Once the benchmark has history, the trajectory adds nothing.** Not a little —
nothing, on both benchmarks, in both scenarios. A new model's per-task outcome is
predictable at 0.87–0.94 AUC from other models' pass rates alone, before its first
token is generated.

That kills early stopping as the lever and replaces it with a better one: **most
(model, task) cells are foregone conclusions, and they are identifiable in advance.**
The savings are in choosing which cells to run, not in interrupting runs. Reading
trajectories only earns its keep on a benchmark nobody has run before — and there,
on SWE-bench Pro, TF-IDF is already at 0.84, so a reasoning model has a high bar to
clear and should be measured against it, not assumed.

What this does *not* yet say: how much cost the prior actually saves at what
ranking-preservation cost. AUC is not a dollar figure. That is step 3.

## Reproduce

```bash
pip install -e .
python -m tally.pull --phase aggregates      # ~400 small files, seconds
python -m tally.pull --phase samples         # ~7.4 GB, resumable
python -m tally.matrix                       # step 1: both sources, the diff, the tables
python -m tally.early extract                # step 2: prefix features + text at N=10
python -m tally.early fit                    # step 2: every AUC above
```

`snapshot_download` does not work on this datastore — the Hub returns an empty
siblings list for a repo this size, so it filters nothing and reports success. The
puller lists each collection through the tree API and fetches per file instead.

## Known limits

- **One scaffold.** Every trajectory is Inspect AI's `react` solver with `bash` and
  `python`. Scaffold choice moves cost up to 33× on identical tasks, so anything
  learned here is scaffold-specific. Rank prediction should transfer; absolute
  numbers will not.
- **Fixed budgets per benchmark** — 10M tokens on Terminal-Bench, 30M on SWE-bench
  Pro — one to three orders of magnitude above typical defaults. Costs here are
  costs under generous budgets.
- **Six models.** The new-model result is leave-one-out over six; a prior from five
  models is what was tested, not a prior from fifty.
- **Every SWE-bench Pro `sample_id` carries a redaction artefact** (`…<AWS-SECRET-KE…>`).
  The 54 remain distinct, so joins work, but never trust the suffix.
- **SWE-bench Pro is a 54-task subset**, not the full benchmark; ~1,700 of its
  attempts come from shards whose condition is unlabelled.

## License

MIT — see [LICENSE](LICENSE).
