# Table 2: matched behavior-cloning comparison

| Environment | Expert mean ± std | BC mean ± std | BC / expert |
|---|---:|---:|---:|
| Ant-v2 | 4713.65 ± 12.20 | 4698.23 ± 88.54 | 99.67% |
| Hopper-v2 | 3772.67 ± 1.95 | 833.16 ± 85.90 | 22.08% |

Both runs meet the requested 30% conditions. Hyperparameters are recorded in results.json and the LaTeX caption.

Reproduce from the repository root:

```bash
bash results/bc_comparison/run.sh
.venv/bin/python scripts/report_bc_comparison.py
```

These are one-seed results. Standard deviations measure variability across evaluation episodes, not across independently trained policies.
The same hidden-layer sizes are used; input/output dimensions and therefore total parameter counts differ by task.
Hopper was selected first: five live expert episodes averaged 3716.14 ± 2.30 versus 3772.67 in saved demonstrations.
Expert check: results/dagger_comparison/expert_check_Hopper/. Walker2d and HalfCheetah were not needed.
To rerun only Hopper while preserving Ant: bash results/bc_comparison/run.sh Hopper

## Event logs and checkpoints

- `hw1/data/q1_table2_bc_Ant_L3_H64_U1000_seed1_Ant-v2_15-09-2026_19-40-22`
- `hw1/data/q1_table2_bc_Hopper_L3_H64_U1000_seed1_Hopper-v2_16-09-2026_20-33-38`
