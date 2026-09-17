# Horizon-averaged BC experiment

| Environment | K | Mean return | Std | Difference from classic BC |
|---|---:|---:|---:|---:|
| Humanoid-v2 | 1 | 305.98 | 51.61 | +0.00 |
| Humanoid-v2 | 2 | 251.76 | 28.90 | -54.22 |
| Humanoid-v2 | 4 | 261.11 | 36.19 | -44.87 |
| Humanoid-v2 | 8 | 238.08 | 10.91 | -67.89 |
| Humanoid-v2 | 16 | 285.41 | 24.35 | -20.56 |

Humanoid-v2: No tested K>1 exceeded the classic BC mean; the hypothesis was not supported in this setup.

Each condition contains 3 independently trained policies, each evaluated on 20 matched reset seeds.
The mean and population std pool the equally sized sets of episode returns. Error bars are episode dispersion, not confidence intervals.
Hidden architecture, demonstrations, batch size, update count, optimizer, learning rate, initialization protocol, and evaluation protocol are fixed. Only K and its required output dimension vary.
This is an equal-update comparison, not an equal-compute comparison. No alternative loss weight is tuned.
