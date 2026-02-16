#!/bin/bash
set -euo pipefail

gpu_id=$1
ckpt_name=$2
seed=${4:-0}  # Default seed is 0

has_tmux=0
if command -v tmux >/dev/null 2>&1; then
  has_tmux=1
fi

run_eval() {
  local k_steps="$1"
  local eval_seed="$2"
  local cmd="conda activate pfp_env && CUDA_VISIBLE_DEVICES=$gpu_id WANDB__SERVICE_WAIT=300 xvfb-run -a python scripts/evaluate.py log_wandb=True env_runner.env_config.vis=False policy.ckpt_name=$ckpt_name seed=$eval_seed policy.num_k_infer=$k_steps"
  if [ "$has_tmux" -eq 1 ]; then
    local session_name="eval_${ckpt_name}_k${k_steps}_${eval_seed}"
    tmux new-session -d -s "$session_name"
    tmux send-keys -t "$session_name" "$cmd" Enter
    echo "[tmux] started $session_name"
  else
    echo "[warn] tmux not found; running in current shell for k=$k_steps seed=$eval_seed"
    bash -lc "$cmd"
  fi
}

for k_steps in 1 2 4 8; do
    if [ "$seed" -ne 0 ]; then
        run_eval "$k_steps" "$seed"
    else
        for eval_seed in 5678 2468 1357; do
            run_eval "$k_steps" "$eval_seed"
        done
        seed=0
    fi
done
