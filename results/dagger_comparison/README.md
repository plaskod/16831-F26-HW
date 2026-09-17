# Figure 2: DAgger

| Environment | BC mean ± std | Final DAgger mean ± std | Expert mean | Final training transitions |
|---|---:|---:|---:|---:|
| Ant-v2 | 4698.23 ± 88.54 | 4787.55 ± 62.31 | 4713.65 | 11555 |
| Hopper-v2 | 833.16 ± 85.90 | 3722.96 ± 3.13 | 3772.67 | 11820 |

Reproduce from the repository root:

```bash
bash results/dagger_comparison/run.sh
.venv/bin/python scripts/report_dagger_comparison.py
```

The report reads the newest matching event directory for each task and the saved Table 2 references.
To rerun only Hopper while preserving Ant: bash results/dagger_comparison/run.sh Hopper
Iteration 0 reproduces Table 2 BC mean/std; iterations 1–9 add expert-labeled learner data.
All ten checkpoints, finite metrics, and collection budgets are checked by the exporter.
This is one training seed; error bars describe evaluation episodes, not uncertainty across training runs.
DAgger uses more data and updates than the fixed BC reference, as requested by the homework.
Evaluation has at least five complete episodes per iteration; episode counts vary with early termination.
The trainer saves aggregate evaluation statistics, not individual evaluation trajectories. Videos are disabled.

Outputs: figure2.png, figure2.svg, figure2.tex (caption), results.json (all iterations/config).
No assignment training code was changed for these runs.

## TensorBoard events and policy checkpoints

- `hw1/data/q2_figure2_dagger_Ant_L3_H64_U1000_seed1_Ant-v2_16-09-2026_19-46-20`
- `hw1/data/q2_figure2_dagger_Hopper_L3_H64_U1000_seed1_Hopper-v2_16-09-2026_20-33-56`
