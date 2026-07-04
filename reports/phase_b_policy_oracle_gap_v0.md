# Phase-B Policy Oracle Gap Report

- TRACER policy: `tracer_proxy_ppo_argmax_v1`
- Source: `reports/phase_b_reach_terminated_policy_comparison_v0.json`

This report compares the current TRACER high-level policy against the best fixed-action baseline per terrain.
The fixed-action oracle is an analysis tool, not a deployable adaptive policy.

| world | TRACER reward | TRACER reach | best fixed policy | best fixed reward | best fixed reach | gap | recommendation |
|---|---:|---:|---|---:|---:|---:|---|
| earth | 7.643 | 1.000 | fixed_trot_solid_fast | 7.622 | 1.000 | 0.022 | keep_current |
| stairs_single | 5.599 | 0.667 | fixed_trot_mid | 7.375 | 1.000 | -1.775 | shift_preference_toward_trot_mid |
| tracer_sponge_firm_flat | 4.681 | 1.000 | fixed_trot_solid_fast | 3.417 | 0.667 | 1.263 | keep_current |

## Notes

- `tracer_sponge_firm_flat`: TRACER reaches the goal neighborhood, but R_s is negative due to post-reach drift. Treat this as frozen low-level hold limitation.
