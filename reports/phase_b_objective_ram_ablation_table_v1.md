# Phase-B Objective/RAM Offline Ablation Table v1

This is an offline counterfactual ablation over directly sampled fixed-command rollouts.
Each selector mode chooses one action per terrain from the same action bank.

## Selector definitions

| mode | meaning |
|---|---|
| Objective OFF / RAM OFF | reach-only selector |
| Objective ON / RAM OFF | reach + final/hold objective, ignores invalid risk |
| Objective OFF / RAM ON | reach objective with invalid/warning penalty |
| Objective ON / RAM ON | task objective + RAM validity penalty |

## Fixed-command action statistics

| world | action | n | reach | deploy_success | hard_invalid | warning | final | drift | score_v1c |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | trot_mid | 30 | 0.167 | 0.167 | 0.067 | 0.067 | 0.200 | 0.015 | 3.180 |
| earth | trot_solid_fast | 30 | 1.000 | 1.000 | 0.000 | 0.000 | 0.171 | 0.024 | 5.153 |
| sponge | sponge_slow_high_clear | 30 | 0.567 | 0.000 | 0.133 | 0.467 | 1.869 | 1.713 | -0.704 |
| sponge | sponge_v1d_bias_late_hold_025_013 | 30 | 0.633 | 0.000 | 0.067 | 0.300 | 2.057 | 1.932 | -0.576 |
| sponge | sponge_v1d_stabilized_late_hold_035_016 | 30 | 0.700 | 0.000 | 0.233 | 0.467 | 1.845 | 1.711 | -0.623 |
| sponge | sponge_v8b_reach_bias | 30 | 0.467 | 0.000 | 0.267 | 0.467 | 1.843 | 1.665 | -1.257 |
| stairs | trot_mid | 30 | 0.933 | 0.800 | 0.167 | 0.033 | 0.251 | 0.138 | 4.313 |
| stairs | trot_solid_fast | 30 | 1.000 | 0.967 | 0.033 | 0.000 | 0.246 | 0.166 | 4.875 |

## Objective/RAM selector ablation

| world | selector mode | selected action | selector score | reach | deploy_success | hard_invalid | warning | final | drift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| earth | Objective OFF / RAM OFF | trot_solid_fast | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.171 | 0.024 |
| earth | Objective OFF / RAM ON | trot_solid_fast | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.171 | 0.024 |
| earth | Objective ON / RAM OFF | trot_solid_fast | 3.920 | 1.000 | 1.000 | 0.000 | 0.000 | 0.171 | 0.024 |
| earth | Objective ON / RAM ON | trot_solid_fast | 3.920 | 1.000 | 1.000 | 0.000 | 0.000 | 0.171 | 0.024 |
| sponge | Objective OFF / RAM OFF | sponge_v1d_stabilized_late_hold_035_016 | 0.700 | 0.700 | 0.000 | 0.233 | 0.467 | 1.845 | 1.711 |
| sponge | Objective OFF / RAM ON | sponge_v1d_bias_late_hold_025_013 | 0.350 | 0.633 | 0.000 | 0.067 | 0.300 | 2.057 | 1.932 |
| sponge | Objective ON / RAM OFF | sponge_v1d_stabilized_late_hold_035_016 | 1.806 | 0.700 | 0.000 | 0.233 | 0.467 | 1.845 | 1.711 |
| sponge | Objective ON / RAM ON | sponge_v1d_bias_late_hold_025_013 | 1.053 | 0.633 | 0.000 | 0.067 | 0.300 | 2.057 | 1.932 |
| stairs | Objective OFF / RAM OFF | trot_solid_fast | 1.000 | 1.000 | 0.967 | 0.033 | 0.000 | 0.246 | 0.166 |
| stairs | Objective OFF / RAM ON | trot_solid_fast | 0.933 | 1.000 | 0.967 | 0.033 | 0.000 | 0.246 | 0.166 |
| stairs | Objective ON / RAM OFF | trot_solid_fast | 3.819 | 1.000 | 0.967 | 0.033 | 0.000 | 0.246 | 0.166 |
| stairs | Objective ON / RAM ON | trot_solid_fast | 3.719 | 1.000 | 0.967 | 0.033 | 0.000 | 0.246 | 0.166 |

## Key interpretation

- Earth and stairs contain deploy-valid fixed commands; high-level selection mainly needs to choose the correct command.
- Sponge has no deploy-valid primitive in the current theta-lite action bank; selector changes can reduce risk but cannot create a valid gait.
- RAM ON tends to penalize hard-invalid/warning-prone choices, which is most important on sponge.
- Objective ON is useful only when the action bank contains physically valid candidates; otherwise it can only choose the least-bad candidate.
