# TRACER Phase-F6 True Metric Beta Targets v0

This converts Phase-F5 true metric objective scores into bootstrap beta target seeds for future Objective Selector calibration.

- input: `datasets/phase_f/f5_true_metric_objective_scores_v0.csv`
- output: `datasets/phase_f/f6_true_metric_beta_targets_v0.csv`
- beta_floor: `0.05`
- rows: `9`

## Beta target seeds

| tag | reset_y | beta_motion | beta_stability | beta_energy | dominant | motion_score | stability_score | energy_score | balanced |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| clean | 0.0 | 0.3498 | 0.3003 | 0.3498 | motion | 1.0000 | 0.8514 | 1.0000 | 0.9509 |
| clean_r1 |  | 0.2516 | 0.3673 | 0.3811 | energy | 0.4473 | 0.6761 | 0.7034 | 0.6073 |
| clean_r2 |  | 0.3865 | 0.3758 | 0.2376 | motion | 0.7168 | 0.6956 | 0.4214 | 0.6123 |
| m030 | -0.3 | 0.0414 | 0.3519 | 0.6067 | energy | 0.0000 | 0.3749 | 0.6825 | 0.3490 |
| m030_r1 |  | 0.3986 | 0.1630 | 0.4383 | energy | 0.6196 | 0.2239 | 0.6862 | 0.5110 |
| m030_r2 |  | 0.1182 | 0.4908 | 0.3910 | stability | 0.1888 | 0.9417 | 0.7399 | 0.6191 |
| p030 | 0.3 | 0.5769 | 0.3820 | 0.0410 | motion | 0.6529 | 0.4155 | 0.0000 | 0.3591 |
| p030_r1 |  | 0.2945 | 0.3787 | 0.3268 | stability | 0.4624 | 0.6090 | 0.5187 | 0.5294 |
| p030_r2 |  | 0.3448 | 0.4635 | 0.1917 | stability | 0.5243 | 0.7221 | 0.2693 | 0.5054 |

## Safe interpretation

This is a true-metric beta target seed table, not a completed IRL Objective Selector. With only three rollout conditions, it should be used for calibration and sanity checking, not for a strong learned model claim.
