#!/usr/bin/env bash
# Bootstrap a Nebius AI Cloud VM (Ubuntu 22.04/24.04) to run Tally for real.
#
# Inference is remote (Token Factory), so the VM only needs CPU, RAM and Docker:
# a 8 vCPU / 32 GB instance is plenty; Terminal-Bench task images want disk,
# so give it 100 GB+. Copy this file to the VM and run it as a sudo-capable
# user. Then copy the repo (see the end) and set NEBIUS_API_KEY.
set -euo pipefail

sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git tmux

# Docker (the sandbox for every trial)
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

# Tally: copy the repo without the 7 GB of samples -- the plan and the
# per-attempt CSVs are all the VM needs.
#   from your machine:
#   scp -r E:/Freelancing/tally/{tally,pyproject.toml,LICENSE,README.md} user@vm:~/tally/
#   scp E:/Freelancing/tally/data/{plan_terminalbench.json,attempts_*.csv,tb2_task_ids.json} user@vm:~/tally/data/
mkdir -p "$HOME/tally/data"
if [ -f "$HOME/tally/pyproject.toml" ]; then
  cd "$HOME/tally"
  uv venv --python 3.13 .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
  pip install -e .
fi

cat <<'EOF'

Next, in a NEW login shell (so the docker group applies):
  export NEBIUS_API_KEY=...            # or write it to ~/tally/.nebius_key
  cd ~/tally && . .venv/bin/activate
  python -m tally.run --plan data/plan_terminalbench.json --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B --limit 1 --attempts 1
Measure tokens per attempt from that one trial before committing to the full plan:
  python -m tally.report --plan data/plan_terminalbench.json
Then, inside tmux so it survives disconnects:
  python -m tally.run --plan data/plan_terminalbench.json --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B --phase all -n 4
EOF
