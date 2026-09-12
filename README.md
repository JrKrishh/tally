# Tally

**Cost-aware evaluation for coding agents. Same ranking, a fraction of the spend, with the receipt.**

Status: **step 1 — data validation.** Nothing is measured yet. This README will
grow only as claims are earned.

## The problem

Evaluating agents has become the compute bottleneck. One pass over SWE-bench
Multimodal runs over $2,200. A statistically credible evaluation — eight reruns per
cell — takes the Holistic Agent Leaderboard from $40K to roughly $320K. τ-bench
scores drop from 60% to 25% once consistency is checked. Static benchmarks solved
this with subsampling (100–200× fewer items, same rankings); agent benchmarks manage
2–3.5×. The people closest to the problem say rigorous evaluation is now gated to
well-funded labs.

## The data

[Every Eval Ever](https://github.com/evaleval/every_eval_ever) hosts a deposited
inference-scaling study on its datastore: six frontier models on Terminal-Bench
(89 tasks) and SWE-bench Pro (54 tasks), multiple attempts per task, every attempt
logged with outcome, turns, tool calls, tokens and latency. Its authors ask people
to reuse it instead of re-running baselines. This project does.

## Step 1: does the data say what the aggregates say?

```bash
pip install -e .
python -m tally.pull --phase aggregates      # ~400 small files
python -m tally.matrix --source aggregates   # matrix inferred from shard metadata
python -m tally.pull --phase samples         # ~7.4 GB, resumable
python -m tally.matrix                       # ground truth, diffed against the inference
```

If the two matrices disagree, the shard metadata was misread and nothing built on
it can be trusted. That check runs before anything else exists.

## License

MIT — see [LICENSE](LICENSE).
