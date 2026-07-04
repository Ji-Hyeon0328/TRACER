# Phase-B Sponge RAM Risk Gate Rationale

This report motivates a post-reach RAM risk gate for `tracer_sponge_firm_flat`.

| policy | reach_rate | final_dist | post_reach_drift | R_v | R_s | reward |
|---|---:|---:|---:|---:|---:|---:|
| raw_ppo | 1.000 | 3.822 | 3.755 | 9.596 | -3.043 | 4.681 |
| fixed_trot_cautious | 0.333 | 0.851 | 0.557 | 4.610 | 0.877 | 3.049 |
| fixed_trot_soft_mid_clear | 0.000 | 0.763 | 0.291 | 1.326 | 1.402 | 1.267 |
| alpha015_selector | 0.667 | 2.713 | 2.501 | 6.733 | -1.537 | 3.487 |

## Learned RAM/Objective shadow

- RAM risk: `0.878`
- RAM uncertainty: `0.534`
- beta_v: `0.143`
- beta_s: `0.754`
- beta_e: `0.104`

## Conclusion

Whole-episode replacement is not ideal for sponge. The better next test is a post-reach RAM risk gate: use the raw PPO action for approach, then switch to a cautious/hold-like action after reaching the goal neighborhood or when high RAM risk is active.
