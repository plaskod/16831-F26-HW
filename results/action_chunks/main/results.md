# Future-action supervision experiment

| Environment | K | Mean return | Episode std, equal seed weight | Seed means | Rollouts per seed |
|---|---:|---:|---:|---|---|
| Ant-v2 | 1 | 4156.99 | 1160.69 | 4689.77, 3971.66, 3809.55 | [5, 6, 6] |
| Ant-v2 | 2 | 4063.96 | 1250.70 | 4563.32, 4580.50, 3048.06 | [5, 5, 5] |
| Ant-v2 | 4 | 3713.27 | 1149.75 | 4328.14, 3976.18, 2835.49 | [5, 5, 5] |
| Ant-v2 | 8 | 1575.72 | 1177.95 | 2872.08, 1095.26, 759.82 | [5, 5, 5] |
| Ant-v2 | 16 | 1003.62 | 465.96 | 1287.68, 754.23, 968.95 | [5, 5, 5] |
| Humanoid-v2 | 1 | 309.84 | 61.44 | 344.12, 254.57, 330.85 | [77, 107, 80] |
| Humanoid-v2 | 2 | 295.99 | 59.45 | 299.39, 284.39, 304.18 | [92, 89, 88] |
| Humanoid-v2 | 4 | 279.71 | 50.40 | 283.00, 231.87, 324.27 | [98, 107, 83] |
| Humanoid-v2 | 8 | 246.23 | 38.98 | 250.41, 250.46, 237.83 | [110, 109, 105] |
| Humanoid-v2 | 16 | 238.84 | 23.72 | 234.88, 261.61, 220.04 | [113, 105, 120] |

Ant-v2: none of K=2,4,8 exceeded the K=1 aggregate mean.
Humanoid-v2: none of K=2,4,8 exceeded the K=1 aggregate mean.

These are descriptive results from 3 training seeds, not a statistical significance test.
Error bars are the population std of episode returns under an equal-weight mixture of seeds:
mean = mean(seed_means); variance = mean(seed_variances + seed_means^2) - mean^2.
This avoids overweighting a seed merely because its failed episodes are shorter.
The figure shows final aggregate means and standard-deviation error bars only; seed means remain in the table.
See aggregate.json for separate standard deviations of the seed means.
