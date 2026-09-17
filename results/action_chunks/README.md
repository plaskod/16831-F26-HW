# One homework experiment: does a longer prediction horizon help BC?

Hypothesis: predicting several future expert actions from the current observation can improve imitation by providing temporal supervision, even when only the first prediction is executed.

## Controlled comparison

The sole varied hyperparameter is K in {1,2,4,8,16}. The loss is MSE averaged over action components and valid horizon offsets. K=1 is ordinary BC. There is no alternative weighted objective. At every environment step, the policy observes the new state, predicts a new chunk, and executes only offset zero.

Default task: Ant-v2, one of the tasks used in Table 2. Each horizon uses:

- The same two supplied trajectories: 2,000 starting observations.
- Three hidden layers of 64 tanh units, linear action output.
- Adam, learning rate 0.005; 1,000 gradient updates, minibatches of 100.
- Training seeds 1,2,3, with identical minibatch sampling protocol.
- Matched hidden initialization AND current-action output weights/biases for each seed.
- Exactly 20 evaluation episodes per trained policy, maximum 1,000 steps each; reset seed `10000 + 1000 * training_seed + episode_index`.
- No video logging and no training-data subsampling or new observation normalization.

Chunks overlap, with one example per starting observation. Future labels never cross terminal or trajectory boundaries. Missing tail labels are masked and the loss averages the valid offsets for each observation, retaining all 2,000 observations. The final output layer must grow with K; hidden layers do not. This is equal-update, not equal-compute, training. The relative weight of the executed action within the averaged loss changes with K; that is part of the stated hypothesis, not an extra tuned parameter.

## Run and plot

From the repository root:

```bash
source .venv/bin/activate
python scripts/run_action_chunk_experiment.py
python scripts/plot_action_chunk_experiment.py
```

The completed results are already in `main/`. The runner refuses to overwrite existing results. To repeat, choose an unused output directory and pass it to both existing scripts:

```bash
python scripts/run_action_chunk_experiment.py --output-dir results/action_chunks/repeat
python scripts/plot_action_chunk_experiment.py results/action_chunks/repeat
```

All five homework tasks remain supported via `--envs`, but the default homework figure uses only Ant to keep the experiment small. `--eval-episodes` controls the fixed episode count. The older step-budget evaluation options have been removed from this standalone runner; the original homework CLI retains its normal evaluation-batch interface.

## Outputs and interpretation

- `main/figure1.png` and `.svg`: one clear final-performance curve with standard-deviation error bars; K=1 is labeled Classic BC.
- `main/figure1.tex`: figure and caption for inclusion in the report (paths assume compilation from `hw1/`). It has not been inserted into the submission.
- `main/results.md` and `aggregate.json`: measured results, differences from baseline, and per-seed means.
- `main/config.json`: exact settings, versions, source hashes.
- Per-condition directories: policy checkpoint, TensorBoard events, 1,000 training losses, and each evaluation episode's rewards and seed.

Each point pools 60 episode returns: 20 episodes for each of three training seeds. Equal episode counts give equal weight to seeds. Error bars show population standard deviation of episode returns, not uncertainty of the mean or a significance test. There are only three independent training runs per horizon.

The initial sweep and diagnostic follow-up were archived under `.local/action_chunks_diagnostics/`, along with their standalone scripts/tests. They are not part of this experiment. They taught us that small mean differences are sensitive to evaluation sampling and that changing loss weighting answers a different question. We therefore keep matched evaluation/initialization as controls, and return to the single original averaged-MSE hypothesis. No diagnostic loss sweep or compute-matching sweep is included.

Run checks with `python -m unittest discover -s tests -v`. Tests cover ordinary BC, chunk targets/masks, boundary handling, sampling, loss gradients, first-action-only execution, and matched initialization.

## Matching Humanoid run

The same existing runner and plotter were used for Humanoid-v2. The Humanoid results are in `humanoid/` and have the same horizons, seeds, demonstrations per task, architecture, updates, loss, and evaluation protocol as Ant. The source hashes and all settings except environment/output directory match.

```bash
python scripts/run_action_chunk_experiment.py --envs Humanoid-v2 --output-dir results/action_chunks/humanoid_repeat
python scripts/plot_action_chunk_experiment.py results/action_chunks/humanoid_repeat
```

`humanoid/figure1.png` is the plot and `humanoid/figure1.tex` is its standalone LaTeX caption. Humanoid does not show improvement over classic BC at any tested horizon under this protocol.
