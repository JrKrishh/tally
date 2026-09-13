# Tally

**Cost-aware evaluation for coding agents. Same ranking, a fraction of the spend, with the receipt.**

Status: **steps 1–5, first real evaluation complete.** The data is validated, the
mechanism is measured, the cost figure is earned in simulation — **Terminal-Bench at
17.7% of the tokens with the six-model ranking intact** — and the product has now run a
real evaluation end to end: Harbor drove NVIDIA Nemotron 3 Nano on Nebius Token Factory
through the cells the plan chose, on a Nebius AI Cloud VM. **Nemotron Nano scores an
estimated 0.090 on Terminal-Bench 2.0, seventh of seven** — every planned task measured
at least once, 120 verdicts, a data point nobody had. Second and third attempts are
pending an API balance that ran out three hours in (step 5).

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
The savings are in choosing which cells to run, not in interrupting runs.

## Step 3: what does selection actually save?

AUC is not a dollar figure. So: each model is treated as new in turn, its task
difficulties come from the other five, only tasks whose difficulty falls inside a
window are run, the rest are imputed from the prior, and attempts per run cell are
optionally capped. Cost and the six-model ranking are compared against the full
evaluation on the tasks all six share; subsampled attempts are averaged over five
seeds.

The full evaluation is **17.8 billion tokens** on Terminal-Bench (86 tasks) and
**25.5 billion** on SWE-bench Pro (40 tasks) — the order of magnitude behind the
$40K–$320K figures above.

| policy | Terminal-Bench cost · Spearman · acc MAE | SWE-bench Pro cost · Spearman · acc MAE |
|---|---|---|
| full evaluation | 100% · 1.000 · 0.000 | 100% · 1.000 · 0.000 |
| skip tasks history calls ≥95% certain | **76% · 1.000 · 0.014** | **52% · 1.000 · 0.017** |
| skip ≥90% certain | 68% · 1.000 · 0.023 | 43% · 0.943 · 0.020 |
| cap at 3 attempts per cell, run every task | 26% · 1.000 · 0.012 | 8.9% · 0.951 · 0.016 |
| skip ≥90% certain **and** cap at 3 | **17.7% · 0.989 · 0.028** | 3.6% · 0.886 · 0.028 |
| skip ≥80% certain and cap at 3 | 13.2% · 0.943 · 0.043 | 2.4% · 0.840 · 0.036 |

Two levers, and they are not equal:

1. **Skipping the cells history calls certain is lossless.** A quarter of Terminal-Bench
   and half of SWE-bench Pro goes unrun, the ranking is untouched, and every model's
   accuracy is within 0.017 of the full run.
2. **Capping attempts is the larger saving, and it is where the trade-off lives.** The
   source study ran 10–15 attempts per cell because it was measuring inference-scaling
   curves; three per cell recovers the ranking at 26% and 9% of the cost. Combined with
   skipping, **Terminal-Bench evaluates at 17.7% of its tokens with Spearman 0.989 and
   the top model preserved in all five seeds — 5.6× cheaper.**

Against which baseline, honestly: 5.6× is against the study's own budget. Against a
sensible three-attempt default, task selection alone buys a further 1.5× on
Terminal-Bench and is not worth it on SWE-bench Pro at 40 tasks.

**The SWE-bench Pro caveat.** Its top two models score 0.810 and 0.805 — a gap of one
task in two hundred, inside the 12% per-attempt flip rate. No policy resolves that
ordering, and neither does the full evaluation; "top-1 kept" there is a coin flip by
construction. Spearman over six models also takes only a handful of values, so read
it alongside the MAE.

## Step 4: cold start — can the task text stand in for history?

A benchmark nobody has run has no history, so the only prior left is the task
itself, read by a reasoning model. This is the one job in the harness that needs
one, and it has a measured ceiling to be judged against. NVIDIA Nemotron 3 Super
(`nvidia/nemotron-3-super-120b-a12b` on Nebius Token Factory) was asked, per task,
zero-shot, for the probability a frontier agent solves it. The kill criterion was
declared before the run: Spearman ≥ 0.3 against true difficulty *and* beats a
text-length baseline.

| | Spearman vs true difficulty | text-length baseline | verdict |
|---|---|---|---|
| Terminal-Bench (85 of 88 parsed) | **0.326** | 0.120 | pass, narrowly |
| SWE-bench Pro (52 of 54 parsed) | 0.133 | −0.381 | **fail** |

Self-contained tasks can be read for difficulty, weakly. Repo-grounded tasks cannot:
the difficulty lives in the codebase the model never sees, and the length of the
issue predicts it better than the model does. Receipt for all 142 tasks: ~98K input
and ~83K output tokens — cents. The model's reasoning arrived in a separate field
averaging 2,800 characters per Terminal-Bench task; at a 300-token cap the answer
field came back *empty with no error*, which cost one wasted run before it was
understood.

The correlation is not what matters. The policy is:

| step-3 policy — skip outside [0.1, 0.9], cap 3 — prior from | Terminal-Bench cost · Spearman · MAE | SWE-bench Pro cost · Spearman · MAE |
|---|---|---|
| history (the other five models) | 17.7% · 0.989 · 0.028 | 3.6% · 0.886 · 0.028 |
| **Nemotron reading the task, no history** | 23.8% · 0.943 · 0.014 | 8.5% · 0.911 · 0.021 |
| no prior at all (cap 3, run every task) | 26.4% · 1.000 · 0.012 | 8.9% · 0.951 · 0.016 |

Read against the right row: **most of the cold-start saving is the attempt cap.** The
model's read of the task buys a further ~10% on Terminal-Bench at a small fidelity
cost, and nothing on SWE-bench Pro. History is 1.3–2.4× cheaper still, because it
skips with confidence. The Nemotron row's higher Spearman on SWE-bench Pro is not a
win over history — it runs twice as many tasks, which is a cost-fidelity trade, not
a better prior.

What the weak prior does have is the property worth keeping: **it fails safe.**
Uncertain estimates land inside the window and the task runs. The harness degrades
from history → weak prior → no prior gracefully, never by skipping a task it
shouldn't; the three Terminal-Bench tasks the model couldn't score were handled
exactly that way.

So the reasoning model's place in this harness is a real, modest contribution where
tasks can be read, an honest zero where they cannot, and both measured against what
it replaces. Untested, and legitimate as *second* experiments if labelled as such:
anchoring with a few scored tasks from another benchmark, pairwise "which is harder"
ranking, and combining the estimate with issue length on SWE-bench Pro.

## Step 5: running it for real

Steps 1–4 are replays of the study's logs. A product plans an evaluation, runs it, and
reports it — on a model nobody has evaluated. That model is Nemotron itself; it is not
in the study.

```
tally plan     history -> which cells to run, how many attempts, what to expect
tally run      Harbor + Terminus-2 execute exactly those cells: the model on Token Factory, one Docker sandbox per trial
tally report   measured where run, history where skipped, rank among the six, the receipt, and a check on the cells NOT run
```

`plan` on Terminal-Bench 2.0 (all 88 study tasks map to TB2 task directories exactly):
57 tasks with uncertain difficulty × 3 attempts, 31 skipped as certain, 8 of those run
once anyway as a check, plus `query-optimize`, which has no history and therefore runs.

**Two real trials** of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` on `adaptive-rejection-sampler`
(prior 0.30): 39 agent steps, 11–12 minutes, verifier 0/9 — while the agent's final
message said the implementation was *"fully developed to meet all specified
requirements."* Harbor's receipt: **616K prompt tokens, 48K completion, 79% of the
prompt served from cache.** ~664K tokens per attempt; the full plan is ~121M, a third
of what frontier-model trajectory lengths implied.

### The validation run, and what it caught

Eight cells history called certain, run once each. 22 minutes.

| history said | tasks | Nemotron Nano actually |
|---|---|---|
| near-certain fail (0.00) | 4 | **failed 4 of 4** |
| near-certain pass (0.91–1.00) | 4 | **passed 1 of 4** |

The fail side transfers: what frontier models cannot do, a 30B model cannot do either.
The pass side does not: "95% certain" was 95% certain *for the frontier tier*. Step 3's
simulation was leave-one-out among six models of the same tier, so it could never have
seen this, and the estimate it would have produced for Nemotron was inflated by 21
near-pass tasks imputed at ~0.95 against a real rate nearer one in four.

That is the whole reason the validate phase exists, and it changes the product: for a
model of unknown tier, **skip only on the fail side** (`plan --hi 1.0`). The pass side
runs until the model has earned its own history. Ten tasks skipped instead of thirty-one;
the saving is smaller and the estimate is honest.

### The full run

The fail-side plan — 78 tasks × 3 attempts plus `query-optimize` — ran on a Nebius AI
Cloud VM (`cpu-d3`, 8 vCPU, 200 GB) with four sandboxes in parallel, Nemotron Nano served
by Token Factory. Four hours wall-clock.

| | |
|---|---|
| trials | 237 launched: 115 clean, 5 agent timeouts (20 min), **117 failed with HTTP 402** |
| coverage | **every one of the 79 planned tasks has at least one verdict** (38 with one, 41 with two) |
| verdicts | 120 — **13 passes across 11 tasks** |
| tokens | 24.3M in, 2.9M out, **18.6M of the input served from cache** |
| cost | **$2.16 at Token Factory list price** ($0.06/M in, $0.24/M out; no cache discount assumed) — plus ~$1 of VM time |
| skipped-cell check | 4 of 4 near-certain fails failed |

**Nemotron 3 Nano 30B on Terminal-Bench 2.0, estimated over all 89 tasks: 0.090 — seventh
of seven**, well below Opus 4's 0.407. That is the first number in this project that came
from running an evaluation rather than replaying one, and it is a data point nobody has
deposited: none of the six frontier models' logs say anything about a 30B model.

The 402s are the Token Factory balance running out three hours in — the one input I
could not verify from outside the console, and the one that failed. They cost nothing
but the second and third attempts, which is why coverage is complete and depth is not.
Two flips already visible in the depth we have: `git-leak-recovery` and `kv-store-grpc`
failed in the validation run and passed here — the 12% per-attempt variance from step 1,
live on a different model.

What it changed: the watchdog now stops the run on the first 402 instead of waiting for
completions to stall (in-flight trials kept finishing for an hour after the balance
died), and `run --job-name … --retry-errored` re-runs exactly the cells that failed on
payment, leaving every verdict untouched.

## Reproduce

```bash
pip install -e .
python -m tally.pull --phase aggregates      # ~400 small files, seconds
python -m tally.pull --phase samples         # ~7.4 GB, resumable
python -m tally.matrix                       # step 1: both sources, the diff, the tables
python -m tally.early extract                # step 2: prefix features + text at N=10
python -m tally.early fit                    # step 2: every AUC above
python -m tally.select                       # step 3: the policy table
python -m tally.coldstart extract            # step 4: task text + true difficulty
NEBIUS_API_KEY=... python -m tally.coldstart score   # step 4: Nemotron on Token Factory, resumable
python -m tally.coldstart report             # step 4: re-analyse scores on disk, spends nothing
python -m tally.plan --lo 0.10 --hi 1.0      # step 5: cells to run for a new model, skipping only the fail side
python -m tally.run --plan data/plan_terminalbench.json --phase validate -n 4   # step 5: check the skipped cells (Docker + NEBIUS_API_KEY)
python -m tally.run --plan data/plan_terminalbench.json --phase run -n 4        # step 5: the real evaluation
python -m tally.report --plan data/plan_terminalbench.json                      # step 5: accuracy, rank, receipt, validation
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
- **Six models.** Every leave-one-out result is a prior from five models, not fifty,
  and a ranking over six.
- **One reasoning model, one prompt, zero-shot.** The cold-start numbers are a floor
  for what task text can give, not a ceiling.
- **Every SWE-bench Pro `sample_id` carries a redaction artefact** (`…<AWS-SECRET-KE…>`).
  The 54 remain distinct, so joins work, but never trust the suffix.
- **SWE-bench Pro is a 54-task subset**, not the full benchmark; ~1,700 of its
  attempts come from shards whose condition is unlabelled.

## License

MIT — see [LICENSE](LICENSE).
