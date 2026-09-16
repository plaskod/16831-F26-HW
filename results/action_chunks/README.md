# Future-action supervision, closed-loop execution

Hypothesis: predicting a moderate number of future expert actions from the current observation improves BC, even when only the first action is executed. A second hypothesis predicts degradation at large horizons. Neither outcome is assumed by the analysis.

## What changed

- `MLP_policy.py`: optional `action_chunk_size` (default 1). Continuous output width is K times the action dimension. `get_action` evaluates the network on every observation and returns only offset zero, with the same action shape as before. No cached actions or open-loop execution.
- `replay_buffer.py`: a separate `ActionChunkReplayBuffer` builds trajectory-local targets and masks. It retains every starting observation and never crosses a terminal or file trajectory boundary. Existing one-step replay is unchanged.
- `bc_agent.py`: selects the new buffer only when K > 1 and passes the horizon to the policy.
- `run_hw1.py`: exposes `--action_chunk_size`. Rejects K > 1 with DAgger: relabeled learner trajectories do not contain consecutive expert-executed actions, so their future labels would answer a different scientific question.

For a complete chunk the objective is mean squared error averaged over K offsets and action dimensions. At an episode tail, it averages only valid offsets for that observation, then averages observations in the batch. Invalid offsets are zero-filled and masked; no fabricated action labels contribute to loss. Thus all 2,000 starting observations remain available at every K. The hidden network is unchanged; the output layer necessarily grows. K=1 uses the original replay buffer, original output shape, and original MSE path.

## Reproduce the complete experiment

From the repository root with `.venv` activated:

```bash
python scripts/run_action_chunk_experiment.py --output-dir results/action_chunks/repeat
python scripts/plot_action_chunk_experiment.py results/action_chunks/repeat
```

Use a new output directory each time. Defaults: Ant-v2 and Humanoid-v2; K=1,2,4,8,16; training seeds 1,2,3; three hidden layers of 64 tanh units; Adam lr=0.005; 1,000 updates; batch size 100. No expert-data subsampling or observation normalization is introduced.

The runner trains using the homework's BCAgent and policy. It evaluates after training, replanning at every environment step, with at least 5 complete episodes and 5,000 transitions per trained policy. Episode lengths are capped at 1,000. Reset seeds are `10000 + 1000 * training_seed + episode_index`, with the same schedule across K. This explicit schedule differs from the earlier Table 2 evaluation, so K=1 returns need not equal Table 2 numbers despite the same learning algorithm. Each environment step consumes just one action vector.

All five HW1 environments are supported, for example:

```bash
python scripts/run_action_chunk_experiment.py --envs Walker2d-v2 Hopper-v2 HalfCheetah-v2 --output-dir results/action_chunks/other_tasks
```

The original homework CLI also supports a single condition (run from `hw1`):

```bash
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --env_name Ant-v2 --exp_name bc_ant_K8 \
  --action_chunk_size 8 --n_iter 1 --n_layers 3 --size 64 \
  --learning_rate 0.005 --num_agent_train_steps_per_iter 1000 \
  --train_batch_size 100 --eval_batch_size 5000 --ep_len 1000 \
  --seed 1 --video_log_freq -1 --no_gpu --save_params
```

Use the dedicated runner for the matched evaluation-seed protocol and per-episode recording.

## Outputs

`main/` is the completed 30-run experiment:

- `figure1.png` and `figure1.svg`: two-panel scientific plot.
- `figure1.tex`: caption and inclusion snippet for `hw1/hw1_submission.tex`; it has not been inserted into the submission. Its image path assumes compilation from `hw1`.
- `results.md` / `aggregate.json`: measured results, seed means, episode counts, and aggregation conventions.
- `config.json`: sweep settings, dependency versions, source hashes.
- Each environment/K/seed folder has `result.json` with all episode rewards, reset seeds, lengths, returns, training losses, dataset hash, and parameter count; `policy.pt`; and TensorBoard events.

```bash
python -m tensorboard.main --logdir results/action_chunks/main
```

The plot uses equal training-seed weights: mu=mean(mu_s), sigma^2=mean(sigma_s^2+mu_s^2)-mu^2. Error bars show episode-return dispersion, not standard error or a confidence interval. Individual seed means are retained in the results table but are not plotted. Episodes from one policy are not independent training replications. No statistical significance claim is made.

## Findings and limits

Neither task's aggregate K=2,4,8 performance exceeded K=1. Longer horizons degraded aggregate performance in this experiment. Thus the proposed moderate-horizon benefit was not supported under this fixed budget; large-horizon degradation was observed. Some individual seeds improved at small horizons. These outcomes should not be generalized to all architectures, optimizers, data budgets, or environments.

This tests future-action supervision without open-loop execution. It does not directly measure representation quality. Output-head size, optimization difficulty, and reducing the relative weight of the first-action loss as K grows remain possible explanations. Masking at trajectory tails changes the number of supervised offsets there. No hyperparameters were tuned separately for K.

## Verification

```bash
python -m unittest discover -s tests -v
```

11 tests passed, covering baseline learning, future-target boundary handling, replay capacity/alignment, masked-loss gradients, same hidden initialization, and first-action-only replanning. A separate check against the pre-change Git version verified bit-for-bit K=1 weights/actions after 10 updates. A K=4 run through the original CLI completed. K=8 training and interaction smoke checks passed for Hopper, Walker2d, and HalfCheetah; Ant/Humanoid were covered by the full sweep.
