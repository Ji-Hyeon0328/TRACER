# TRACER Phase-I0 RAM Teacher Dataset v0

This builds heuristic RAM teacher targets for supervised pretraining.

- input: `datasets/phase_h/h0_objective_selector_training_table_v0.csv`
- output csv: `datasets/phase_i/i0_ram_teacher_dataset_v0.csv`
- manifest: `datasets/phase_i/i0_ram_teacher_manifest_v0.json`
- rows: `22`
- input features: `23`

## Rows by base condition

| base_tag | rows |
|---|---:|
| clean | 3 |
| m015 | 3 |
| m030 | 3 |
| m060 | 3 |
| p015 | 3 |
| p030 | 3 |
| p045 | 1 |
| p060 | 3 |

## Target summary

| target | mean | std | min | max |
|---|---:|---:|---:|---:|
| rho_slip_target | 0.454269 | 0.278133 | 0.165886 | 1.000000 |
| rho_rough_target | 0.458988 | 0.273479 | 0.016807 | 1.000000 |
| sigma_target | 0.518588 | 0.143241 | 0.311525 | 0.809808 |

## Per-condition target means

| base_tag | n | rho_slip | rho_rough | sigma |
|---|---:|---:|---:|---:|
| clean | 3 | 0.311022 | 0.191033 | 0.371334 |
| m015 | 3 | 0.542005 | 0.548206 | 0.576806 |
| m030 | 3 | 0.687439 | 0.344649 | 0.556856 |
| m060 | 3 | 0.553661 | 0.678954 | 0.541672 |
| p015 | 3 | 0.281323 | 0.251521 | 0.441326 |
| p030 | 3 | 0.268351 | 0.488113 | 0.471798 |
| p045 | 1 | 0.271490 | 0.838933 | 0.803165 |
| p060 | 3 | 0.597007 | 0.583795 | 0.575464 |

## Safe interpretation

These are heuristic RAM teacher seeds derived from tracking/lateral/contact/energy proxies. They are useful for RAM supervised pretraining and runtime plumbing, but they are not yet a final teacher-student RAM learned from real proprioceptive prediction error.
