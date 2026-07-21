# TRACER Phase-F5 True Metric Objective Scores v0

This converts Phase-F4 true rollout metrics into normalized objective scores for Objective Selector / RAM follow-up training.

- input: `datasets/phase_f/f4_true_metric_training_table_v0.csv`
- output: `datasets/phase_f/f5_true_metric_objective_scores_v0.csv`
- rows: `9`

## Ranking by balanced score

| rank | tag | reset_y | motion | stability | energy | balanced | vx | lat_abs | power | imu |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | clean | 0.0 | 1.0000 | 0.8514 | 1.0000 | 0.9509 | 0.1752 | 0.0363 | 155.0666 | 0.4370 |
| 2 | m030_r2 |  | 0.1888 | 0.9417 | 0.7399 | 0.6191 | 0.1729 | 0.0359 | 157.3889 | 0.4220 |
| 3 | clean_r2 |  | 0.7168 | 0.6956 | 0.4214 | 0.6123 | 0.1744 | 0.0372 | 160.2333 | 0.4433 |
| 4 | clean_r1 |  | 0.4473 | 0.6761 | 0.7034 | 0.6073 | 0.1737 | 0.0361 | 157.7148 | 0.4649 |
| 5 | p030_r1 |  | 0.4624 | 0.6090 | 0.5187 | 0.5294 | 0.1737 | 0.0362 | 159.3646 | 0.4525 |
| 6 | m030_r1 |  | 0.6196 | 0.2239 | 0.6862 | 0.5110 | 0.1742 | 0.0397 | 157.8687 | 0.4517 |
| 7 | p030_r2 |  | 0.5243 | 0.7221 | 0.2693 | 0.5054 | 0.1739 | 0.0364 | 161.5918 | 0.4440 |
| 8 | p030 | 0.3 | 0.6529 | 0.4155 | 0.0000 | 0.3591 | 0.1743 | 0.0373 | 163.9965 | 0.4796 |
| 9 | m030 | -0.3 | 0.0000 | 0.3749 | 0.6825 | 0.3490 | 0.1724 | 0.0389 | 157.9014 | 0.4607 |

## Safe interpretation

This is a bootstrap true-metric objective score table, not yet a full IRL Objective Selector. It is suitable for calibrating beta/objective targets and checking whether physical proxies agree with previous deployment choices.
