# Horizon-averaged BC experiment

| Environment | K | Mean return | Std | Difference from classic BC |
|---|---:|---:|---:|---:|
| Ant-v2 | 1 | 4147.58 | 1129.64 | +0.00 |
| Ant-v2 | 2 | 4347.96 | 722.36 | +200.38 |
| Ant-v2 | 4 | 3110.75 | 1572.67 | -1036.83 |
| Ant-v2 | 8 | 1058.86 | 678.30 | -3088.71 |
| Ant-v2 | 16 | 1270.10 | 976.69 | -2877.47 |

Ant-v2: K=2 exceeded the classic BC mean, providing limited descriptive support for the hypothesis; this is not a significance result.

Each condition contains 3 independently trained policies, each evaluated on 20 matched reset seeds.
The mean and population std pool the equally sized sets of episode returns. Error bars are episode dispersion, not confidence intervals.
Hidden architecture, demonstrations, batch size, update count, optimizer, learning rate, initialization protocol, and evaluation protocol are fixed. Only K and its required output dimension vary.
This is an equal-update comparison, not an equal-compute comparison. No alternative loss weight is tuned.
