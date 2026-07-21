# TRACER Phase-F6 True Metric Beta Targets v0

This converts Phase-F5 true metric objective scores into bootstrap beta target seeds for future Objective Selector calibration.

- input: `datasets/phase_f/f5_true_metric_objective_scores_v0.csv`
- output: `datasets/phase_f/f6_true_metric_beta_targets_v0.csv`
- beta_floor: `0.05`
- rows: `3`

## Beta target seeds

| tag | reset_y | beta_motion | beta_stability | beta_energy | dominant | motion_score | stability_score | energy_score | balanced |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| clean | 0.0 | 0.3333 | 0.3333 | 0.3333 | motion | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| m030 | -0.3 | 0.0441 | 0.3104 | 0.6456 | energy | 0.0000 | 0.3022 | 0.6825 | 0.3250 |
| p030 | 0.3 | 0.6549 | 0.2985 | 0.0466 | motion | 0.6529 | 0.2704 | 0.0000 | 0.3112 |

## Safe interpretation

This is a true-metric beta target seed table, not a completed IRL Objective Selector. With only three rollout conditions, it should be used for calibration and sanity checking, not for a strong learned model claim.
