#!/usr/bin/env bash
# From the laptop (Git Bash): bring the VM's results home and report on them.
#
#   bash deploy/pull.sh ubuntu@VM_IP
#
# Fetches only what report needs -- result.json, trajectory.json, reward.txt,
# config.json -- and leaves the asciinema recordings (hundreds of MB) on the VM.
# Grab those separately for the video: scp -r user@vm:~/tally/jobs/<job>/<trial>/agent .
set -euo pipefail
VM="${1:?usage: bash deploy/pull.sh user@ip}"
cd "$(dirname "$0")/.."

ssh "$VM" 'cd ~/tally && find jobs -type f \( -name result.json -o -name trajectory.json -o -name reward.txt -o -name config.json \) | tar czf /tmp/tally-results.tgz -T -'
scp "$VM:/tmp/tally-results.tgz" ./
tar xzf tally-results.tgz
rm -f tally-results.tgz
python -m tally.report --plan data/plan_terminalbench.json
