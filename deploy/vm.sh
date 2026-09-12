#!/usr/bin/env bash
# Bootstrap an Ubuntu 22.04/24.04 VM to run Tally's real evaluations overnight.
#
# Inference is remote (Token Factory), so the box needs CPU, RAM, disk and Docker:
# 8+ vCPU, 32 GB RAM, 100 GB+ disk (Terminal-Bench task images add up). Each
# trial is a Docker container, so the host MUST be a real VM:
#
#   Nebius AI Cloud  Create resource -> Virtual machine -> platform cpu-d3 or cpu-e2,
#                    Ubuntu image, 100 GB+ boot disk. Cheapest, and it is the
#                    hackathon's own compute.
#   Vast.ai          ONLY a "VM" offer (KVM, systemd). A standard Vast instance is
#                    itself a container and does not allow Docker inside it; Harbor
#                    fails at the first trial there. Filter offers by VM support,
#                    pick an Ubuntu VM image, 8+ vCPU / 32 GB / 100 GB.
#
# Copy this file to the VM, run it as a sudo-capable user, then copy the repo and
# set the key (see the printed steps).
set -euo pipefail

sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git tmux

# Docker: the sandbox for every trial
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER"

# uv, Python 3.13 (Harbor's pin), Harbor
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.13
uv tool install harbor --python 3.13
harbor --version

# Tally, if the repo has already been copied (see below)
mkdir -p "$HOME/tally/data"
if [ -f "$HOME/tally/pyproject.toml" ]; then
  cd "$HOME/tally"
  uv venv --python 3.13 .venv
  uv pip install --python .venv/bin/python -e .     # uv venvs ship without pip
fi

cat <<'EOF'

== From your machine: copy the repo WITHOUT the 7 GB of samples ==
  scp -r E:/Freelancing/tally/tally E:/Freelancing/tally/pyproject.toml E:/Freelancing/tally/LICENSE user@vm:~/tally/
  scp E:/Freelancing/tally/data/plan_terminalbench.json E:/Freelancing/tally/data/attempts_terminalbench.csv E:/Freelancing/tally/data/tb2_task_ids.json user@vm:~/tally/data/
  (re-run this script once the repo is there so the venv gets created)

== On the VM, in a NEW login shell so the docker group applies ==
  export NEBIUS_API_KEY=...                 # or write it to ~/tally/.nebius_key
  cd ~/tally && . .venv/bin/activate
  python -m tally.run --plan data/plan_terminalbench.json --limit 1 --attempts 1    # one trial: proves the box
  python -m tally.report --plan data/plan_terminalbench.json
  tmux new -s tally                                                                 # survives disconnects
  python -m tally.run --plan data/plan_terminalbench.json --phase run -n 4          # -n 8 on 16 vCPU
  # detach: Ctrl-b d    reattach: tmux attach -t tally

== Afterwards: bring the results home (result files only; .cast recordings are large) ==
  rsync -avz --include='*/' --include='result.json' --include='trajectory.json' --include='reward.txt' --exclude='*' user@vm:~/tally/jobs/ E:/Freelancing/tally/jobs/
  python -m tally.report --plan data/plan_terminalbench.json
EOF
