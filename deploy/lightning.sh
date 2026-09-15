#!/usr/bin/env bash
# Run a model's default plan on the VM, one phase at a time, each under the watchdog:
# validate (every task history calls certain, once), then run (uncertain tasks, plan attempts).
#
#   tmux new-session -d -s lrun "bash ~/tally/deploy/lightning.sh [model] [plan]"
#
# Progress and phase boundaries go to ~/tally/lightning.log; the watchdog kills the
# lrun session on a 402/401 or an error streak, and says why in jobs/watchdog.log.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/tally" && . .venv/bin/activate
MODEL="${1:-nvidia/Nemotron-3_5-Lightning}"
PLAN="${2:-data/plan_terminalbench_lightning.json}"
SLUG=$(echo "$MODEL" | sed 's/[^A-Za-z0-9]\{1,\}/-/g; s/^-//; s/-$//')
LOG="$HOME/tally/lightning.log"

for phase in validate run; do
  pattern="jobs/tally-terminalbench-$phase-$SLUG-*"
  before=$(ls -d $pattern 2>/dev/null | wc -l)
  python -m tally.run --plan "$PLAN" --model "$MODEL" --phase "$phase" -n 4 > "lightning-$phase.log" 2>&1 &
  pid=$!
  job=""
  for _ in $(seq 1 60); do
    if [ "$(ls -d $pattern 2>/dev/null | wc -l)" -gt "$before" ]; then
      job=$(ls -td $pattern | head -1)
      break
    fi
    sleep 10
  done
  echo "$(date -Is) $phase started: ${job:-no job dir yet}" >> "$LOG"
  [ -n "$job" ] && tmux new-session -d -s "lwatch-$phase" "bash $HOME/tally/deploy/watchdog.sh $HOME/tally/$job lrun 600"
  wait "$pid"; rc=$?
  tmux kill-session -t "lwatch-$phase" 2>/dev/null
  echo "$(date -Is) $phase finished rc=$rc" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "$(date -Is) STOPPED after $phase" >> "$LOG"
    exit "$rc"
  fi
done
echo "$(date -Is) ALL-DONE" >> "$LOG"
