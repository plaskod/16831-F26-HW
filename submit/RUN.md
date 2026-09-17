# HW1 reproduction

Andrew ID: dplaskow. Generative AI collaborator: GPT-6 Astra.

## Setup

Run all commands from this folder (the folder containing `rob831`). Use Python
3.10 and the course MuJoCo installation instructions. The reported experiments
used Python 3.10.18, CPU, Gym 0.25.1, mujoco-py 2.1.2.14 and the MuJoCo 2.1.1
runtime on Apple Silicon macOS.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```


## Regenerate tables and figures from submitted logs

```bash
python scripts/expert_stats.py
python scripts/report_bc_comparison.py
python scripts/report_dagger_comparison.py
python scripts/plot_action_chunk_experiment.py \
  run_logs/q1_action_chunks_ant run_logs/q1_action_chunks_hopper \
  --output-dir results/action_chunks/ant_hopper
```

## Table 1: expert demonstrations

`scripts/expert_stats.py` sums rewards in each of the two supplied trajectories
for each of the five environments.

## Table 2: run BC

```bash
for robot in Ant Hopper; do
  python rob831/scripts/run_hw1.py \
    --expert_policy_file "rob831/policies/experts/${robot}.pkl" \
    --expert_data "rob831/expert_data/expert_data_${robot}-v2.pkl" \
    --env_name "${robot}-v2" \
    --exp_name "table2_bc_${robot}_L3_H64_U1000_seed1" \
    --n_iter 1 --action_chunk_size 1 \
    --n_layers 3 --size 64 --learning_rate 0.005 \
    --num_agent_train_steps_per_iter 1000 --train_batch_size 100 \
    --eval_batch_size 5000 --ep_len 1000 --seed 1 \
    --video_log_freq -1 --scalar_log_freq 1 --no_gpu --save_params
done
```

## Figure 2: run DAgger

```bash
for robot in Ant Hopper; do
  python rob831/scripts/run_hw1.py \
    --expert_policy_file "rob831/policies/experts/${robot}.pkl" \
    --expert_data "rob831/expert_data/expert_data_${robot}-v2.pkl" \
    --env_name "${robot}-v2" \
    --exp_name "figure2_dagger_${robot}_L3_H64_U1000_seed1" \
    --do_dagger --n_iter 10 --action_chunk_size 1 \
    --n_layers 3 --size 64 --learning_rate 0.005 \
    --num_agent_train_steps_per_iter 1000 --train_batch_size 100 \
    --batch_size 1000 --eval_batch_size 5000 --ep_len 1000 --seed 1 \
    --video_log_freq -1 --scalar_log_freq 1 --no_gpu --save_params
done
```


```bash
cp -Rp data/q1_table2_bc_* run_logs/
cp -Rp data/q2_figure2_dagger_* run_logs/
python scripts/report_bc_comparison.py
python scripts/report_dagger_comparison.py
```

## Figure 1: run the action-prediction-horizon experiment

```bash
python scripts/run_action_chunk_experiment.py \
  --envs Ant-v2 Hopper-v2 --horizons 1 2 4 8 16 --seeds 1 2 3 \
  --updates 1000 --train-batch-size 100 --n-layers 3 --size 64 \
  --learning-rate 0.005 --eval-episodes 20 --ep-len 1000 \
  --output-dir results/action_chunks/rerun
python scripts/plot_action_chunk_experiment.py results/action_chunks/rerun
```

