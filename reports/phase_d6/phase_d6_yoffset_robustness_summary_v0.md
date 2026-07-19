# TRACER Phase-D6.3b Lateral Offset Robustness Summary v0

## Scope

This report aggregates Phase-D6 MLP-gated selector control under lateral reset offsets.

The validated runtime structure is:

```text
MLP selector -> gate -> /tracer/mpc_reference

learned selected dimensions:
  vx=True
  yaw=True
  clearance=True

empirical/safety protected dimensions:
  body_h=False
  enable=False
  goal/hold phase = empirical fallback
```

## Offset-level summary

| tag | reset_y | n | success_rate | goal_rate | startup_failed_rate | out_lane_rate | hold_drift_mean | summary_json |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| m030 | -0.30 | 3.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.027667 | `reports/phase_d6/phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| p000 | 0.00 | 3.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.083667 | `reports/phase_d6/phase_d4_context_meta_repeat_20260719_183716_d6_mlp_gated_yoffset_p000_n3_summary_v0.json` |
| p030 | 0.30 | 3.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.028333 | `reports/phase_d6/phase_d4_context_meta_repeat_20260719_185241_d6_mlp_gated_yoffset_p030_n3_summary_v0.json` |

## Individual rollouts

| tag | reset_y | trial | success | goal | startup_failed | out_lane | final_x | final_y | max_abs_y | mean_abs_y | hold_drift | final_context |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
| m030 | -0.30 | 1 | True | True | False | False | 8.113000 | -0.820000 | 0.820000 | 0.417836 | 0.000000 | goal_flat |
| m030 | -0.30 | 2 | True | True | False | False | 8.174000 | -0.014000 | 0.072000 | 0.025848 | 0.007000 | goal_flat |
| m030 | -0.30 | 3 | True | True | False | False | 8.000000 | -1.036000 | 1.036000 | 0.451272 | 0.076000 | goal_flat |
| p000 | 0.00 | 1 | True | True | False | False | 7.996000 | -0.062000 | 0.076000 | 0.026410 | 0.067000 | goal_flat |
| p000 | 0.00 | 2 | True | True | False | False | 7.869000 | -0.196000 | 0.210000 | 0.045047 | 0.184000 | goal_flat |
| p000 | 0.00 | 3 | True | True | False | False | 8.226000 | 0.124000 | 0.245000 | 0.122370 | 0.000000 | goal_flat |
| p030 | 0.30 | 1 | True | True | False | False | 8.250000 | -0.316000 | 0.432000 | 0.227078 | 0.000000 | goal_flat |
| p030 | 0.30 | 2 | True | True | False | False | 8.173000 | -0.168000 | 0.168000 | 0.052859 | 0.024000 | goal_flat |
| p030 | 0.30 | 3 | True | True | False | False | 7.989000 | -0.675000 | 0.689000 | 0.358318 | 0.061000 | goal_flat |

## Interpretation

Phase-D6.3b checks whether the MLP-gated selector remains valid when the robot is laterally perturbed at reset.

A successful result means the controller is not only memorizing the near-center initial condition. It can still reach the goal while remaining inside the lateral bound under moderate initial y-offset perturbations.

This remains a supervised learned selector with a safety gate, not a full RL meta-planner.

