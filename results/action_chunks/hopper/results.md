# Horizon-averaged BC experiment

| Environment | K | Mean return | Std | Difference from classic BC |
|---|---:|---:|---:|---:|
| Hopper-v2 | 1 | 789.25 | 218.28 | +0.00 |
| Hopper-v2 | 2 | 841.90 | 223.75 | +52.65 |
| Hopper-v2 | 4 | 962.79 | 195.22 | +173.54 |
| Hopper-v2 | 8 | 582.70 | 367.96 | -206.56 |
| Hopper-v2 | 16 | 289.07 | 155.52 | -500.19 |

Hopper-v2: K=2, 4 exceeded the classic BC mean, providing limited descriptive support for the hypothesis; this is not a significance result.

Each condition contains 3 independently trained policies, each evaluated on 20 matched reset seeds.
The mean and population std pool the equally sized sets of episode returns. Error bars are episode dispersion, not confidence intervals.
Hidden architecture, demonstrations, batch size, update count, optimizer, learning rate, initialization protocol, and evaluation protocol are fixed. Only K and its required output dimension vary.
This is an equal-update comparison, not an equal-compute comparison. No alternative loss weight is tuned.
