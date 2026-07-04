# Phase-B Learned Objective/RAM Shadow Report

Shadow mode only: learned Objective/RAM outputs are recorded but do not affect action selection or the frozen low-level controller.

| world | n | actions | reach_rate | beta_v | beta_s | beta_e | recovery | RAM risk | RAM uncertainty | reach_reward | hold_reward | final_dist |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 3 | trot_mid | 1.000 | 0.087 | 0.864 | 0.048 | 0.000 | 0.000 | 0.006 | 6.779 | 1.456 | 0.187 |
| stairs_single | 3 | trot_solid_fast | 0.667 | 0.091 | 0.859 | 0.050 | 0.000 |  |  | 6.371 | -0.054 | 1.008 |
| tracer_sponge_firm_flat | 3 | trot_soft_mid_clear | 1.000 | 0.144 | 0.757 | 0.099 | 0.000 | 0.878 | 0.534 | 8.104 | -7.693 | 3.822 |

## Notes

- If sponge shows high RAM risk/uncertainty while still reaching the goal, this supports the reach/hold separation story.
- If Objective β emphasizes stability on risky terrain, it can later be used to condition action scoring.
- This report does not claim improvement yet; it validates learned-module observability.
