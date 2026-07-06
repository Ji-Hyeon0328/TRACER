# Phase-B Validity-Aware Action Score v1b

Formula:

`2*reach + 0.20*R_v + 1.00*R_s - 3*hard_invalid - 0.75*warning - 0.70*drift - 0.30*final_dist`

| rank | world::action | n | reach | final | drift | R_v | R_s | hard_invalid | warning | score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | tracer_sponge_firm_flat::sponge_v8b_reach_bias | 5 | 0.600 | 1.081 | 0.866 | 6.474 | 0.515 | 0.400 | 0.400 | 0.579 |
| 2 | tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016 | 5 | 0.800 | 1.713 | 1.537 | 7.747 | -0.499 | 0.200 | 0.400 | 0.161 |
| 3 | tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013 | 5 | 0.800 | 2.022 | 1.934 | 8.671 | -1.207 | 0.000 | 0.200 | 0.016 |
| 4 | tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.600 | 1.952 | 1.840 | 7.575 | -0.994 | 0.200 | 0.600 | -1.203 |
