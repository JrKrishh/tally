#!/usr/bin/env bash
# Watch a Harbor job on the VM and stop it if trials start failing in a streak.
#
#   bash deploy/watchdog.sh <jobs_dir/job_name> [tmux_session=tally] [interval_s=600]
#
# An unattended run has one failure mode Harbor will not save you from: an
# exhausted API balance, a revoked key, or a broken daemon makes every trial
# error in seconds, and errored trials are not retried on resume. This loop
# reads the job's result.json every interval and, if errored trials rose by
# eight or more while completed trials did not move, kills the run's tmux
# session and writes why to watchdog.log. Nothing else; it never restarts.
set -uo pipefail
JOB="${1:?usage: watchdog.sh <job_dir> [tmux_session] [interval_s]}"
SESSION="${2:-tally}"
INTERVAL="${3:-600}"
LOG="$(dirname "$JOB")/watchdog.log"
prev_done=-1; prev_err=-1

read_stats() {
  python3 - "$JOB/result.json" <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1])).get("stats", {})
    print(s.get("n_completed_trials", 0), s.get("n_errored_trials", 0), s.get("n_pending_trials", 0))
except Exception:
    print(-1, -1, -1)
EOF
}

echo "$(date -Is) watchdog on $JOB, session $SESSION, every ${INTERVAL}s" >> "$LOG"
while true; do
  sleep "$INTERVAL"
  read -r done err pending <<< "$(read_stats)"
  echo "$(date -Is) completed=$done errored=$err pending=$pending" >> "$LOG"
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "$(date -Is) session $SESSION is gone; watchdog exiting" >> "$LOG"; exit 0
  fi
  # Definitive: any trial that just failed with 402/401 means the balance or key is gone.
  # In-flight trials keep completing for a while, so the streak rule below is too slow here.
  # /dev/null keeps grep off stdin when no result changed recently -- inside tmux that
  # stdin is a terminal and grep would block forever, silently killing the watchdog.
  if grep -lsE '"exception_message": "[^"]*(402|401|Payment Required|exhausted your budget)' /dev/null \
       $(find "$JOB" -mindepth 2 -maxdepth 2 -name result.json -mmin -$((INTERVAL / 60 + 1)) 2>/dev/null) 2>/dev/null | head -1 | grep -q .; then
    echo "$(date -Is) STOPPING: a trial failed with 402/401 -- API balance exhausted or key revoked. Add funds, then: tally.run --job-name $(basename "$JOB") --retry-errored" >> "$LOG"
    tmux kill-session -t "$SESSION"
    exit 1
  fi
  if [ "$done" -ge 0 ] && [ "$prev_err" -ge 0 ] && [ $((err - prev_err)) -ge 8 ] && [ "$done" -eq "$prev_done" ]; then
    echo "$(date -Is) STOPPING: $((err - prev_err)) new errors, no new completions in ${INTERVAL}s -- key, balance or daemon" >> "$LOG"
    tmux kill-session -t "$SESSION"
    exit 1
  fi
  prev_done=$done; prev_err=$err
done
