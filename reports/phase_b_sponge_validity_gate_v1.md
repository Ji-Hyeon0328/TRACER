# Phase-B Sponge Validity Gate v1

- Deploy-valid available: `False`
- Recovery needed: `True`
- Recommended deploy action: `None`
- Reach-safe probe: `tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013`
- Low-drift risky probe: `tracer_sponge_firm_flat::sponge_v8b_reach_bias`

No deploy-valid sponge primitive found in the current frozen low-level theta-lite action bank. Use reach-safe probe only for data collection, not as a deployable traversal primitive.

| action | reach | hard | warning | final | drift | deploy_success | score_v1c | deploy_valid | reach_safe_probe | low_drift_risky |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 0.600 | 0.400 | 0.400 | 1.081 | 0.866 | 0.000 | 0.581 | False | False | True |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016 | 0.800 | 0.200 | 0.400 | 1.713 | 1.537 | 0.000 | 0.118 | False | False | False |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013 | 0.800 | 0.000 | 0.200 | 2.022 | 1.934 | 0.000 | 0.094 | False | True | False |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 0.600 | 0.200 | 0.600 | 1.952 | 1.840 | 0.000 | -1.100 | False | False | False |
