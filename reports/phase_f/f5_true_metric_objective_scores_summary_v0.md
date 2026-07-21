# TRACER Phase-F5 True Metric Objective Scores v0

This converts Phase-F4 true rollout metrics into normalized objective scores for Objective Selector / RAM follow-up training.

- input: `datasets/phase_f/f4_true_metric_training_table_v0.csv`
- output: `datasets/phase_f/f5_true_metric_objective_scores_v0.csv`
- rows: `3`

## Ranking by balanced score

| rank | tag | reset_y | motion | stability | energy | balanced | vx | lat_abs | power | imu |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | clean | 0.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1752 | 0.0363 | 155.0666 | 0.4370 |
| 2 | m030 | -0.3 | 0.0000 | 0.3022 | 0.6825 | 0.3250 | 0.1724 | 0.0389 | 157.9014 | 0.4607 |
| 3 | p030 | 0.3 | 0.6529 | 0.2704 | 0.0000 | 0.3112 | 0.1743 | 0.0373 | 163.9965 | 0.4796 |

## Safe interpretation

This is a bootstrap true-metric objective score table, not yet a full IRL Objective Selector. It is suitable for calibrating beta/objective targets and checking whether physical proxies agree with previous deployment choices.
