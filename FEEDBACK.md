# Feedback from building Tally on Nebius Token Factory and NVIDIA Nemotron

Everything below was hit while doing real work, is reproducible, and cost time.
Ordered by how much time each one cost.

## Nebius Token Factory

**1. A reasoning model that runs out of `max_tokens` mid-thought returns empty content and no error.**
`nvidia/nemotron-3-super-120b-a12b` puts its reasoning in `reasoning_content`. With
`max_tokens=300`, 6 of 10 replies had `content: ""`, `finish_reason` gave no hint, and
HTTP 200. A client that reads only `content` sees a successful call that said nothing.
Cost: one full 142-task scoring run had to be discarded. Suggestion: a `finish_reason`
of `length` when reasoning consumed the budget, or a documented note that reasoning
counts against `max_tokens` and content may be empty.

**2. `usage` does not report reasoning tokens.** `completion_tokens_details.reasoning_tokens`
is absent, so there is no way to see how much of a bill was thinking. The receipt this
harness prints for every run has a blank where that number should be.

**3. Model ids are not the slugs the rest of the ecosystem uses, and a wrong one is a bare 404.**
OpenRouter and NVIDIA call it `nvidia/nemotron-3-nano-30b-a3b`; Token Factory serves it
as `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` while the Super model *is* lowercase
(`nvidia/nemotron-3-super-120b-a12b`). The 404 body names the model but suggests nothing.
Cost: one 11-minute Terminal-Bench trial, three retries, zero result. Suggestion: accept
case-insensitive ids, or return the nearest served id in the error.

**4. Prefix caching works and is not documented anywhere I could find.** Agent loops
against Token Factory reported 79% of prompt tokens as `cached_tokens` (523,776 of
616,347 on one Terminal-Bench trial). This is the single biggest cost lever for agentic
workloads, public commentary claims Nebius has no caching, and the pricing page returns
404 from outside the console — so nobody knows whether cached tokens are discounted.
Suggestion: document it, price it, and put it on the model card.

**5a. There is no way to read the balance from the API, and no low-balance warning.**
A four-hour unattended evaluation hit `402 Payment Required — You have exhausted your
budget` three hours in. Nothing before that moment — no header, no usage endpoint, no
email — said the balance was low. 117 trials failed on payment; because the run was
concurrent, in-flight trials kept completing for an hour after the balance died, so a
watchdog keyed on "nothing completes any more" never fired. Suggestion: a balance/usage
endpoint (even a `X-Balance-Remaining` header on responses), and an alert threshold in
the console. Cost: a night's second and third attempts across 79 tasks.

**5. The pricing and model docs are not reachable without a login.** `docs.tokenfactory.nebius.com/pricing`,
`nebius.com/prices-ai-studio` and the catalog page all 404 or render empty outside the
console. Cost estimates for this project had to be done from OpenRouter's numbers.

## NVIDIA Nemotron

**6. Nemotron 3 Nano's self-report disagrees with the verifier.** On `adaptive-rejection-sampler`
(Terminal-Bench 2.0) its final message was *"the implementation has been fully developed to
meet all specified requirements"*; the verifier failed all nine tests, two with
`FileNotFoundError` on files it said it had created. Not a Nebius issue, but worth knowing
when the model is the agent: the transcript is not evidence.

**7. Zero-shot task-difficulty estimation is weak on repo-grounded tasks.** Asked for
`p_solve` per task, Nemotron 3 Super reached Spearman 0.33 against true difficulty on
Terminal-Bench (self-contained task text) and 0.13 on SWE-bench Pro (issue text plus a
repo it cannot see), where issue length alone did better at −0.38. A reasoning model
reading a task is not a substitute for a look at the codebase.

## Harbor (the Terminal-Bench harness — not a Nebius product, but on the path)

**8. Trial results are written with `Path.write_text()` and no encoding.** On Windows that
is cp1252; the first stored model message containing `≈` raised `UnicodeEncodeError` at
the final write of a finished 11-minute trial and lost its summary while everything under
the trial directory survived. Workaround: `PYTHONUTF8=1`. Fix: `encoding="utf-8"` in
`harbor/trial/trial.py:470`.

**9. `--dry-run` requires a running Docker daemon**, though it downloads nothing and runs
nothing. It cannot validate a config on a machine without Docker.

## Every Eval Ever (the data source)

**10. `snapshot_download` silently matches nothing on the datastore.** The Hub returns an
empty siblings list for a repository this size, so `allow_patterns` filters nothing and
the call reports success in one second with zero files. Workaround: the per-collection
tree API plus `hf_hub_download`.

**11. Shard aggregates undercount attempts 2–3×.** `source_data.sample_ids` plus the
`+Nep` epoch count implies a matrix that the samples contradict on every (model, task)
pair — adaptive sampling makes `+Nep` a ceiling, and eight shards carry 1,600 rows with
no task list. Anyone counting from aggregates gets a third of the data.
