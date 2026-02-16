#!/bin/bash
set -euo pipefail

gpu_id=$1
ckpt_name=$2
k_steps=${3:-50}  # Default k_steps is 50
seed=${4:-0}  # Default seed is 0
render=${5:-False}  # Default render is False

has_tmux=0
if command -v tmux >/dev/null 2>&1; then
  has_tmux=1
fi

run_eval() {
  local eval_seed="$1"
  local cmd="conda activate pfp_env && CUDA_VISIBLE_DEVICES=$gpu_id WANDB__SERVICE_WAIT=300 xvfb-run -a python scripts/evaluate.py log_wandb=True env_runner.env_config.vis=$render policy.ckpt_name=$ckpt_name seed=$eval_seed policy.num_k_infer=$k_steps"
  if [ "$has_tmux" -eq 1 ]; then
    local session_name="eval_${ckpt_name}_k${k_steps}_${eval_seed}"
    tmux new-session -d -s "$session_name"
    tmux send-keys -t "$session_name" "$cmd" Enter
    echo "[tmux] started $session_name"
  else
    echo "[warn] tmux not found; running in current shell for seed=$eval_seed"
    bash -lc "$cmd"
  fi
}

if [ "$seed" -ne 0 ]; then
  run_eval "$seed"
else
  for eval_seed in 5678 2468 1357; do
    run_eval "$eval_seed"
  done
fi
