# Tally

**Cost-aware evaluation for coding agents. Same ranking, a fraction of the spend, with the receipt.**

**Live demo: https://jrkrishh.github.io/tally/**: the planner running on the real data, the Nemotron run's results and receipt, and the layer experiment.

Status: **steps 1–6: the first real evaluation, and one agent layer tested on held-out
tasks.** The data is validated, the mechanism is measured, the cost figure is earned in
simulation against a no-history baseline — **Terminal-Bench at 14.7% of the tokens with
the six-model ranking intact (Spearman 0.982), most of it from capping attempts** — and
the product has now run a real evaluation end to end: Harbor drove NVIDIA Nemotron
3 Nano on Nebius Token Factory through the cells the plan chose, on a Nebius AI Cloud
VM. **Nemotron 3 Nano scores an estimated 0.079 on Terminal-Bench 2.0 (95% interval
0.035–0.131), last of five against the frontier models' runs without correctness
feedback on the same tasks** — every planned task attempted three times, 237 verdicts,
$3.92 of inference, and a data point nobody had. Of the 13 tasks it ever solved, it
solved 12 only some of the time (step 5). **A verify-before-done layer around the agent
does not lift it:** tuned on 40 tasks and run once on 39 held-out tasks, it changed the
pass rate by −0.009 (95% interval −0.051 to +0.034) at 1.8× the cost per trial (step 6).

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
difficulties come from the other five, tasks whose difficulty falls inside a window
get their attempts, and the rest are either imputed from the prior or run once.
Attempts per task are optionally capped. Cost and the six-model ranking are compared
against the full evaluation on the tasks all six share; subsampled attempts are
averaged over 200 seeds.

The full evaluation is **17.8 billion tokens** on Terminal-Bench (86 tasks) and
**25.5 billion** on SWE-bench Pro (40 tasks) — the order of magnitude behind the
$40K–$320K figures above.

| policy | Terminal-Bench cost · Spearman · acc MAE · top-1 kept | SWE-bench Pro cost · Spearman · acc MAE · top-1 kept |
|---|---|---|
| full evaluation | 100% · 1.000 · 0.000 · 100% | 100% · 1.000 · 0.000 · 100% |
| skip tasks history calls ≥95% certain, every attempt | 76% · 1.000 · 0.014 · 100% | 52% · 1.000 · 0.017 · 100% |
| skip ≥90% certain, every attempt | 68% · 1.000 · 0.023 · 100% | 43% · 0.943 · 0.020 · 0% |
| no history: every task, 3 attempts | 26.2% · 0.990 · 0.012 · 100% | 8.9% · 0.947 · 0.016 · 60% |
| no history: every task, 2 attempts | 17.5% · 0.986 · 0.016 · 98% | 5.9% · 0.911 · 0.021 · 51% |
| no history: every task, 1 attempt | 8.8% · 0.973 · 0.023 · 93% | 3.0% · 0.842 · 0.031 · 41% |
| skip ≥90% certain, 3 attempts on the rest | 17.7% · 0.973 · 0.027 · 100% | **3.6% · 0.921 · 0.027 · 42%** |
| **run ≥90%-certain tasks once, 2 attempts on the rest** | **14.7% · 0.982 · 0.018 · 98%** | 4.2% · 0.888 · 0.023 · 48% |
| run ≥90%-certain tasks once, 3 attempts on the rest | 20.5% · 0.987 · 0.015 · 100% | 5.4% · 0.915 · 0.019 · 55% |

Three findings, in order of size:

1. **Capping attempts is the saving, and it needs no history.** The source study ran
   10–15 attempts per cell because it was measuring inference-scaling curves. Two
   attempts per task, with no history at all, keep Terminal-Bench's ranking at Spearman
   0.986 for 17.5% of the tokens.
2. **Skipping what history calls certain is nearly free when every attempt is kept, and
   it is where the error comes from under a cap.** A skipped task costs nothing and is
   wrong by the gap between this model and the average of the others. At three
   attempts on Terminal-Bench that more than doubles the accuracy error (0.012 → 0.027)
   and drops the ranking below the plan that uses no history.
3. **History's job is allocation, not imputation.** Run the certain tasks once and save
   repeat attempts for the uncertain ones: **Terminal-Bench at 14.7% of its tokens,
   Spearman 0.982, the top model kept in 98% of samples** — 6.8× cheaper than the
   study's budget, and 16% cheaper than running every task twice at nearly the same
   fidelity. On SWE-bench Pro, where attempts are long and tasks few, skipping still
   pays: 3.6% of the tokens for 0.921, against 0.842 for one attempt per task at 3.0%.

**Against which baseline, honestly.** The first version of this table reported "skip
≥90% certain and cap at 3" as **17.7% · 0.989**. That was the mean of five attempt
samples; over 200 it is 0.973, and running every task twice with no history beats it at
the same cost. That comparison is the test the result has to pass, so it is in the
table, and `tally plan` now runs every certain task once by default. Most of the saving
on Terminal-Bench is the attempt cap; history buys the last 16%.

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
| history (the other five models) | 17.7% · 0.973 · 0.027 | 3.6% · 0.921 · 0.027 |
| **Nemotron reading the task, no history** | 23.8% · 0.972 · 0.015 | 8.3% · 0.883 · 0.028 |
| no prior at all (cap 3, run every task) | 26.2% · 0.990 · 0.012 | 8.9% · 0.947 · 0.016 |

This table holds the skip-and-impute policy fixed so the priors can be compared; step 3
shows that running certain tasks once beats skipping them. Read against the right row:
**most of the cold-start saving is the attempt cap.** The model's read of the task buys
a further ~9% on Terminal-Bench at a small fidelity cost, and on SWE-bench Pro a 7%
saving that costs more fidelity than it is worth, as the failed kill criterion
predicts. History is 1.3–2.3× cheaper still, because it skips with confidence.

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
the saving is smaller and the estimate is honest. Step 3's 200-sample replay later reached
the same place from inside the frontier tier: imputing certain tasks costs accuracy even
there, so `plan` now runs every certain task once by default and imputes nothing.

### The full run

The fail-side plan — 78 tasks × 3 attempts plus `query-optimize` — ran on a Nebius AI
Cloud VM (`cpu-d3`, 8 vCPU, 200 GB) with four sandboxes in parallel, Nemotron Nano served
by Token Factory. Four hours wall-clock.

| | |
|---|---|
| trials | **237 — every one of the 79 planned tasks attempted 3 times**, 0 infrastructure errors |
| verdicts | 229 clean, 8 agent timeouts (20 min, scored as fails) |
| passes | **21 across 13 tasks** |
| tokens | 42.8M in, 5.6M out, **32.4M of the input served from cache** |
| cost | **$3.92 at Token Factory list price** ($0.06/M in, $0.24/M out; no cache discount assumed) plus $1.35 of VM time for the 5.9 h of trials |
| skipped-cell check | 4 of 4 near-certain fails failed |

**Nemotron 3 Nano 30B on Terminal-Bench 2.0, estimated over all 89 tasks: 0.079, with a
95% interval of 0.035–0.131** (bootstrap over tasks and over attempts within each task).
It is the first number in this project that came from running an evaluation rather than
replaying one, and a data point nobody has deposited — none of the six frontier models'
logs say anything about a 30B model.

**Ranking it against the frontier needs one correction.** Half of the study's attempts
ran with oracle correctness feedback — the agent was told whether its answer was right —
and that roughly doubles the weaker models (Opus 4 goes from 0.233 to 0.473 with it).
Nemotron Nano never had that signal, so an earlier version of this README ranked it
against inflated numbers. Like for like, using only attempts without feedback, on the 57
tasks every included model shares:

| model | accuracy, same 57 tasks, no feedback |
|---|---|
| GPT-5.4 | 0.596 |
| GPT-5.2 | 0.496 |
| Opus 4.6 | 0.487 |
| Opus 4 | 0.187 |
| **Nemotron 3 Nano 30B** (this run) | **0.042** (95% interval 0.006–0.089) |

Last of five, and Opus 4 is still well outside its interval. Opus 4.5 and GPT-5 are left
out: they have only 2 and 11 tasks without feedback. Budgets still differ — 10M-token
trajectories for the frontier models, 60 turns for Nemotron Nano — and the stock agent
threw away 18% of Nano's turns (691 of 3,908): the complete answer came back in
`reasoning_content` with `content` empty, and Terminus-2 reads only `content`. So this is
a floor for the small model, not a verdict on it. (The 2K thinking cap passed with
`--max-thinking` never applied either: Harbor sends it only to Anthropic models.)
`tally.report` and the demo page both compute the comparison this way.

**The consistency finding is sharper than the score.** Of the 79 tasks, 66 failed all
three attempts and exactly one — `prove-plus-comm` — passed all three. The other **12
tasks Nemotron Nano solved, it solved only some of the time**: 12 of its 13 successes do
not reproduce reliably. A single-run leaderboard would report a model that can do those
things; three attempts show a model that sometimes does. `git-leak-recovery` and
`kv-store-grpc` failed in the validation run, passed in the full run, and it is exactly
this kind of task that the extra attempts exist to catch.

It took two sittings. Three hours into the first night the Token Factory balance ran out
and 117 trials failed with HTTP 402 — the one input that could not be checked from outside
the console. Nothing measured was lost: the watchdog now stops on the first 402 (in-flight
trials had kept completing for an hour after the balance died, so a stall detector never
fired), and after funds were added `run --job-name … --retry-errored` deleted exactly the
payment-failed trial directories and Harbor re-ran those 117 cells the next morning in
2 h 40 m for $1.76, every earlier verdict untouched.

## Step 6: can a layer lift a small model?

Nemotron Nano ended 223 of its 237 trials by declaring the task complete, and 202 of those
claims failed the hidden tests. That points at a fix that needs no training: a layer around
the agent that won't take "done" without evidence. It was tested the way a claim should be —
tuned on one half of the tasks, reported on the other, with the halves split before a single
trajectory was read.

**The split.** `python -m tally.boost split` drew it once from the 79 tasks the stock agent
ran, stratified by whether the stock agent ever solved the task, and it was committed before
any tuning (0f0bd04): 40 dev tasks with a stock pass rate of 0.100, 39 held-out at 0.077.

**The layer** (`tally/checked.py`, a Terminus-2 subclass that Harbor loads by import path)
does two things.

1. *It reads answers the stock agent threw away.* The stock trajectories held a loss that
   happened before the model did anything wrong: 703 of 3,908 turns (18%) came back with
   `content` empty and the JSON answer in `reasoning_content`. They are the turns where
   Nano answered without thinking — median 273 completion tokens against 797, and in 245 of
   them the "reasoning" opens with `{` or a code fence, which 7 of 3,183 normal turns do.
   Terminus-2 reads only `content`, so each one became a parse error. The layer takes the
   last complete answer from the reasoning instead, and never an outline with `...` in its
   commands.
2. *It checks "done" before accepting it.* On the first completion claim, a separate Nano
   call writes up to six shell checks from the task text and a directory listing — never the
   agent's work, the hidden tests (Harbor uploads those after the agent stops) or any
   solution. They run in the task container; if any fails, the agent gets the failures with
   their output and keeps working, up to two rejections.

**Dev: +0.056, 95% interval −0.056 to +0.167** (36 of 40 tasks; four multi-GB images never
started on the laptop). Two tasks gained, three lost, and two faults of the kind a tuning run
exists to find: a recovered outline (`cat > plus_comm.v <<'EOF'...EOF`) left the shell
inside a heredoc for the rest of a task the stock agent solves every time, and of the 13
final answers the checks passed, 11 failed the hidden tests.

**Who should write the checks?** `tally.checkeval` measures that without any agent. On each
dev task it runs every writer's checks on the untouched task, then again after Terminal-Bench's
reference solution; a useful set of checks rejects the first and accepts the second.

| writer | prompt | accepts a correct solution | rejects the untouched task | wrong checks | $ per task |
|---|---|---|---|---|---|
| Nemotron 3 Nano 30B | original | **29%** | 100% | 35% | 0.0010 |
| Nemotron 3 Nano 30B | stricter | 24% | 88% | 38% | 0.0011 |
| Nemotron 3 Super 120B | stricter | 18% | 94% | 44% | 0.0052 |
| Nemotron 3 Ultra 550B | stricter | 18% | 100% | 37% | 0.0091 |

These are the 34 dev tasks whose reference solution passes its own tests. On the same prompt,
the bigger models did worse: they write longer shell, and it breaks on details — a `subject=`
compared against an `issuer=` prefix, OpenSSL 3's key output, fingerprint colons. The
stricter prompt, written after dev, hurt every writer, so held-out kept the original. A
threshold doesn't rescue the rule either (`tally.boost thresholds`): rejecting only when half
the checks fail accepts 62% of correct solutions but catches 37% of Nano's wrong final answers,
against 29% and 59% for rejecting on any failure. With 88% of Nano's claims false, "any"
stayed.

**Held-out, run once**, on the same Nebius VM that ran the stock baseline:

| 39 tasks × 3 attempts | stock agent | with layer |
|---|---|---|
| pass rate | 0.077 (9 of 117) | 0.068 (8 of 116) |
| "done" claims that were false | 92% | 92% |
| turns per trial | 16.3 | 21.3 |
| cost per trial | $0.015 | $0.028 |
| timeouts | 7 | 6 |

**Paired difference −0.009, 95% interval −0.051 to +0.034. The layer does not lift Nemotron
Nano.** It issued 172 rejections, and when its checks said fail the hidden tests agreed 69 of
75 times — yet the rejections bought two new tasks (`merge-diff-arc-agi-task`,
`nginx-request-logging`) and lost two (`cancel-async-tasks`, `custom-memory-heap-crash`). The
checks point at real failures; Nano mostly cannot repair them. The distance from 0.04 to the
frontier's 0.5–0.6 is the model, not the scaffold. The misfiled answers remain a platform fix
worth making ([FEEDBACK.md](FEEDBACK.md) item 1). The whole experiment cost $1.11 of inference
on dev, $0.63 for the check writers and $3.20 on held-out.

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
python -m tally.plan --lo 0.10 --hi 1.0 --attempts 3 --validate 8   # step 5: the plan the real run used (the default runs every certain task once)
python -m tally.run --plan data/plan_terminalbench.json --phase validate -n 4   # step 5: check the skipped cells (Docker + NEBIUS_API_KEY)
python -m tally.run --plan data/plan_terminalbench.json --phase run -n 4        # step 5: the real evaluation
python -m tally.report --plan data/plan_terminalbench.json                      # step 5: accuracy, rank, receipt, validation
python -m tally.boost misfiled                                                  # step 6: answers filed as reasoning in the stock run
python -m tally.run --plan data/plan_terminalbench.json --split dev --agent tally.checked:CheckedTerminus --ak max_rejections=2 -n 4
python -m tally.run --plan data/plan_terminalbench.json --split dev --agent tally.checkeval:CheckEval --ak writers=nano-v1,nano,super,ultra --agent-timeout-multiplier 4 -n 3
python -m tally.boost checkeval jobs/<checkeval job>                            # step 6: the writer table
python -m tally.boost thresholds jobs/<checkeval job> jobs/<dev job>            # step 6: any failure vs a share of failures
python -m tally.run --plan data/plan_terminalbench.json --split heldout --attempts 3 --agent tally.checked:CheckedTerminus --ak max_rejections=2 -n 4
python -m tally.boost compare jobs/<held-out job> --table                       # step 6: the paired result
.venv-harbor/Scripts/python tests/test_checked.py                               # the layer's tests, in Harbor's Python
python -m tally.site                                                            # rebuild docs/data.js for the demo page
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
- **One layer, one small model.** Step 6 tests one design on Nemotron Nano. A model that
  repairs what the checks flag could gain from the same layer; this one did not. The dev run
  was on a laptop and the held-out run on the VM, so only held-out is comparable to the
  baseline.

## License

MIT — see [LICENSE](LICENSE).
