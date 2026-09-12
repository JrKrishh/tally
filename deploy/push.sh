#!/usr/bin/env bash
# From the laptop (Git Bash): send everything a fresh VM needs and run the bootstrap.
#
#   bash deploy/push.sh ubuntu@VM_IP
#
# Copies the bootstrap, the repo WITHOUT the 7 GB of samples and without the
# Windows Harbor venv, the plan, the history CSV (report needs it for the ranking)
# and the task-id list; then runs deploy/vm.sh on the VM. Afterwards ssh in, set
# NEBIUS_API_KEY, and follow the steps the bootstrap prints.
set -euo pipefail
VM="${1:?usage: bash deploy/push.sh user@ip}"
cd "$(dirname "$0")/.."

ssh "$VM" 'mkdir -p ~/tally/data ~/tally/deploy'
scp deploy/vm.sh "$VM:~/tally/deploy/"
scp -r tally pyproject.toml LICENSE README.md "$VM:~/tally/"
scp data/plan_terminalbench.json data/attempts_terminalbench.csv data/tb2_task_ids.json "$VM:~/tally/data/"
ssh "$VM" 'rm -rf ~/tally/tally/__pycache__; bash ~/tally/deploy/vm.sh'

cat <<EOF

Pushed and bootstrapped. Next:
  ssh $VM
  export NEBIUS_API_KEY=...
  cd ~/tally && . .venv/bin/activate
  python -m tally.run --plan data/plan_terminalbench.json --limit 1 --attempts 1   # proves the box (~12 min)
  python -m tally.report --plan data/plan_terminalbench.json
  tmux new -s tally
  python -m tally.run --plan data/plan_terminalbench.json --phase run -n 4          # -n 8 on 16 vCPU
  (Ctrl-b d to detach; tmux attach -t tally to return)
When it finishes:  bash deploy/pull.sh $VM
EOF
