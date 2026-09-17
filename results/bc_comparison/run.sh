#!/bin/bash
set -euo pipefail
cd hw1
robots=("$@")
if [ "$#" -eq 0 ]; then robots=(Ant Hopper); fi
for robot in "${robots[@]}"; do
  ../.venv/bin/python rob831/scripts/run_hw1.py \
    --expert_policy_file "rob831/policies/experts/${robot}.pkl" \
    --expert_data "rob831/expert_data/expert_data_${robot}-v2.pkl" \
    --env_name "${robot}-v2" --exp_name "table2_bc_${robot}_L3_H64_U1000_seed1" \
    --n_iter 1 --n_layers 3 --size 64 --learning_rate 0.005 \
    --num_agent_train_steps_per_iter 1000 --train_batch_size 100 \
    --eval_batch_size 5000 --ep_len 1000 --seed 1 \
    --video_log_freq -1 --no_gpu --save_params \
    > "../results/bc_comparison/${robot}.log" 2>&1
done
