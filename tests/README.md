# HW1 implementation checks

From the repository root, with the environment activated:

```bash
python -m unittest discover -s tests -v
```

The tests check network depth, supervised loss reduction, batched/single actions, discrete actions, episode termination and truncation, trajectory counts, replay alignment/capacity, initial expert loading, and DAgger data aggregation.

Short integration runs also passed with these commands from `hw1`:

```bash
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name todo_smoke_bc --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --n_layers 2 --size 64 --num_agent_train_steps_per_iter 100 --train_batch_size 100 --eval_batch_size 200 --ep_len 100 --video_log_freq -1 --no_gpu --save_params

python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name todo_smoke_dagger --n_iter 2 --do_dagger --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl --n_layers 2 --size 64 --num_agent_train_steps_per_iter 100 --train_batch_size 100 --batch_size 200 --eval_batch_size 200 --ep_len 100 --video_log_freq 1 --no_gpu --save_params
```

These are functional checks, not the full performance experiments for the report. The runs are in `hw1/data/`; terminal logs are in `.local/todo-*.log`. Checkpoint tensors and TensorBoard scalar/video events were read back successfully.

Continuous actions are deterministic network predictions, trained with mean-squared error against expert labels. Abstract base-class methods and the intentionally non-trainable expert policy retain their `NotImplementedError` guards.
