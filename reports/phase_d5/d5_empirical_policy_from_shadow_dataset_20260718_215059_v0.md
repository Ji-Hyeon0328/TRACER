# TRACER Phase-D5 Empirical Policy Export v0

- source_dataset: `datasets/phase_d5/phase_d5_shadow_dataset_20260718_215059_dataset_v0.csv`
- out_json: `models/phase_d5/d5_empirical_policy_from_shadow_dataset_20260718_215059_v0.json`
- hold_vx: `0.025`
- yaw_gain: `-0.024996`
- yaw_k: `0.024996`
- yaw_sign: `-1.0`
- yaw_max: `0.2`

## Context action table

| context | n | vx_mean | vx_std | body_h | clearance |
|---|---:|---:|---:|---:|---:|
| flat | 128 | 0.2100 | 0.000000 | 0.320 | 0.045 |
| start_flat | 128 | 0.2100 | 0.000000 | 0.320 | 0.045 |
| upslope | 134 | 0.2100 | 0.000000 | 0.320 | 0.045 |
| rough | 130 | 0.2050 | 0.000000 | 0.320 | 0.055 |
| downslope | 148 | 0.2025 | 0.000000 | 0.320 | 0.045 |
| goal_flat | 3 | 0.2025 | 0.000000 | 0.320 | 0.045 |
| unknown | 3 | 0.2025 | 0.000000 | 0.320 | 0.045 |

## Action table string

```text
flat:0.2100,0.320,0.045;start_flat:0.2100,0.320,0.045;upslope:0.2100,0.320,0.045;rough:0.2050,0.320,0.055;downslope:0.2025,0.320,0.045;goal_flat:0.2025,0.320,0.045;unknown:0.2025,0.320,0.045
```
