# TRACER Phase-D7.0b Objective Bootstrap Inspection v0

- input_csv: `datasets/phase_d7/d7_objective_bootstrap_dataset_v0.csv`
- rows: `115`

## Beta normalization check

- beta_sum_min: `1.000000000`
- beta_sum_max: `1.000000000`
- beta_sum_mean: `1.000000000`

## Context-level means

| context | n | beta_m | beta_s | beta_e | motion_score | stability_score | energy_score | max_abs_y | hold_drift | effort_proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| downslope | 23 | 0.187 | 0.559 | 0.254 | 0.992 | 0.832 | 0.515 | 0.387 | 0.037 | 0.485 |
| flat | 23 | 0.350 | 0.264 | 0.386 | 0.998 | 0.891 | 0.507 | 0.387 | 0.037 | 0.493 |
| goal_flat | 23 | 0.207 | 0.512 | 0.281 | 0.869 | 0.830 | 0.847 | 0.387 | 0.037 | 0.153 |
| rough | 23 | 0.184 | 0.500 | 0.317 | 0.995 | 0.849 | 0.412 | 0.387 | 0.037 | 0.588 |
| upslope | 23 | 0.269 | 0.388 | 0.343 | 0.998 | 0.875 | 0.504 | 0.387 | 0.037 | 0.496 |

## Reset-y-level means

| reset_y | n | beta_m | beta_s | beta_e | max_abs_y | hold_drift |
|---:|---:|---:|---:|---:|---:|---:|
| -0.60 | 5 | 0.210 | 0.522 | 0.269 | 0.718 | 0.356 |
| -0.30 | 15 | 0.230 | 0.467 | 0.303 | 0.643 | 0.028 |
| 0.00 | 60 | 0.245 | 0.431 | 0.324 | 0.262 | 0.027 |
| 0.30 | 30 | 0.238 | 0.448 | 0.314 | 0.437 | 0.014 |
| 0.60 | 5 | 0.237 | 0.449 | 0.314 | 0.490 | 0.000 |

## Highest hold drift samples

| rank | context | reset_y | trial | value | beta_m | beta_s | beta_e | max_abs_y | hold_drift | final_x | final_y | source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | downslope | -0.60 | 1 | 0.356 | 0.162 | 0.628 | 0.210 | 0.718 | 0.356 | 7.725 | 0.231 | `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json` |
| 2 | flat | -0.60 | 1 | 0.356 | 0.309 | 0.358 | 0.333 | 0.718 | 0.356 | 7.725 | 0.231 | `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json` |
| 3 | goal_flat | -0.60 | 1 | 0.356 | 0.173 | 0.599 | 0.228 | 0.718 | 0.356 | 7.725 | 0.231 | `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json` |
| 4 | rough | -0.60 | 1 | 0.356 | 0.166 | 0.558 | 0.276 | 0.718 | 0.356 | 7.725 | 0.231 | `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json` |
| 5 | upslope | -0.60 | 1 | 0.356 | 0.238 | 0.466 | 0.296 | 0.718 | 0.356 | 7.725 | 0.231 | `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json` |
| 6 | downslope | 0.00 | 2 | 0.184 | 0.189 | 0.560 | 0.251 | 0.210 | 0.184 | 7.869 | -0.196 | `phase_d4_context_meta_repeat_20260719_183716_d6_mlp_gated_yoffset_p000_n3_summary_v0.json` |
| 7 | flat | 0.00 | 2 | 0.184 | 0.334 | 0.301 | 0.365 | 0.210 | 0.184 | 7.869 | -0.196 | `phase_d4_context_meta_repeat_20260719_183716_d6_mlp_gated_yoffset_p000_n3_summary_v0.json` |
| 8 | goal_flat | 0.00 | 2 | 0.184 | 0.204 | 0.524 | 0.273 | 0.210 | 0.184 | 7.869 | -0.196 | `phase_d4_context_meta_repeat_20260719_183716_d6_mlp_gated_yoffset_p000_n3_summary_v0.json` |
| 9 | rough | 0.00 | 2 | 0.184 | 0.184 | 0.507 | 0.309 | 0.210 | 0.184 | 7.869 | -0.196 | `phase_d4_context_meta_repeat_20260719_183716_d6_mlp_gated_yoffset_p000_n3_summary_v0.json` |
| 10 | upslope | 0.00 | 2 | 0.184 | 0.258 | 0.416 | 0.325 | 0.210 | 0.184 | 7.869 | -0.196 | `phase_d4_context_meta_repeat_20260719_183716_d6_mlp_gated_yoffset_p000_n3_summary_v0.json` |
| 11 | downslope | -0.30 | 3 | 0.076 | 0.157 | 0.632 | 0.211 | 1.036 | 0.076 | 8.000 | -1.036 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 12 | flat | -0.30 | 3 | 0.076 | 0.321 | 0.324 | 0.355 | 1.036 | 0.076 | 8.000 | -1.036 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |

## Highest lateral deviation samples

| rank | context | reset_y | trial | value | beta_m | beta_s | beta_e | max_abs_y | hold_drift | final_x | final_y | source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | downslope | -0.30 | 3 | 1.036 | 0.157 | 0.632 | 0.211 | 1.036 | 0.076 | 8.000 | -1.036 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 2 | flat | -0.30 | 3 | 1.036 | 0.321 | 0.324 | 0.355 | 1.036 | 0.076 | 8.000 | -1.036 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 3 | goal_flat | -0.30 | 3 | 1.036 | 0.164 | 0.613 | 0.223 | 1.036 | 0.076 | 8.000 | -1.036 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 4 | rough | -0.30 | 3 | 1.036 | 0.159 | 0.564 | 0.277 | 1.036 | 0.076 | 8.000 | -1.036 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 5 | upslope | -0.30 | 3 | 1.036 | 0.247 | 0.437 | 0.316 | 1.036 | 0.076 | 8.000 | -1.036 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 6 | downslope | -0.30 | 1 | 0.820 | 0.167 | 0.608 | 0.225 | 0.820 | 0.000 | 8.113 | -0.820 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 7 | flat | -0.30 | 1 | 0.820 | 0.337 | 0.290 | 0.373 | 0.820 | 0.000 | 8.113 | -0.820 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 8 | goal_flat | -0.30 | 1 | 0.820 | 0.178 | 0.578 | 0.244 | 0.820 | 0.000 | 8.113 | -0.820 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 9 | rough | -0.30 | 1 | 0.820 | 0.170 | 0.534 | 0.296 | 0.820 | 0.000 | 8.113 | -0.820 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 10 | upslope | -0.30 | 1 | 0.820 | 0.253 | 0.422 | 0.325 | 0.820 | 0.000 | 8.113 | -0.820 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 11 | downslope | -0.60 | 1 | 0.718 | 0.162 | 0.628 | 0.210 | 0.718 | 0.356 | 7.725 | 0.231 | `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json` |
| 12 | flat | -0.60 | 1 | 0.718 | 0.309 | 0.358 | 0.333 | 0.718 | 0.356 | 7.725 | 0.231 | `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json` |

## Highest energy beta samples

| rank | context | reset_y | trial | value | beta_m | beta_s | beta_e | max_abs_y | hold_drift | final_x | final_y | source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | flat | -0.30 | 2 | 0.407 | 0.369 | 0.224 | 0.407 | 0.072 | 0.007 | 8.174 | -0.014 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 2 | flat | 0.00 | 2 | 0.402 | 0.363 | 0.236 | 0.402 | 0.163 | 0.005 | 8.154 | -0.110 | `phase_d4_context_meta_repeat_20260719_165843_d6_mlp_gated_robustness_n5_summary_v0.json` |
| 3 | flat | 0.00 | 4 | 0.398 | 0.361 | 0.241 | 0.398 | 0.155 | 0.018 | 8.139 | -0.068 | `phase_d4_context_meta_repeat_20260719_165843_d6_mlp_gated_robustness_n5_summary_v0.json` |
| 4 | flat | 0.00 | 1 | 0.398 | 0.361 | 0.241 | 0.398 | 0.143 | 0.025 | 8.033 | 0.026 | `phase_d4_context_meta_repeat_20260719_162757_d6_mlp_gated_control_smoke_n1_summary_v0.json` |
| 5 | flat | 0.00 | 3 | 0.398 | 0.361 | 0.241 | 0.398 | 0.249 | 0.000 | 8.209 | 0.061 | `phase_d4_context_meta_repeat_20260719_163540_d6_mlp_gated_control_final_n3_summary_v0.json` |
| 6 | flat | 0.00 | 5 | 0.395 | 0.358 | 0.247 | 0.395 | 0.336 | 0.000 | 8.182 | 0.334 | `phase_d4_context_meta_repeat_20260719_165843_d6_mlp_gated_robustness_n5_summary_v0.json` |
| 7 | flat | 0.30 | 1 | 0.395 | 0.357 | 0.248 | 0.395 | 0.354 | 0.000 | 8.180 | 0.137 | `phase_d4_context_meta_repeat_20260719_180101_d6_mlp_gated_yoffset_p030_smoke_n1_retry_summary_v0.json` |
| 8 | flat | 0.00 | 1 | 0.395 | 0.359 | 0.246 | 0.395 | 0.076 | 0.067 | 7.996 | -0.062 | `phase_d4_context_meta_repeat_20260719_183716_d6_mlp_gated_yoffset_p000_n3_summary_v0.json` |
| 9 | flat | 0.30 | 2 | 0.394 | 0.357 | 0.249 | 0.394 | 0.168 | 0.024 | 8.173 | -0.168 | `phase_d4_context_meta_repeat_20260719_185241_d6_mlp_gated_yoffset_p030_n3_summary_v0.json` |
| 10 | flat | 0.00 | 3 | 0.393 | 0.355 | 0.251 | 0.393 | 0.326 | 0.000 | 8.280 | -0.302 | `phase_d4_context_meta_repeat_20260719_165843_d6_mlp_gated_robustness_n5_summary_v0.json` |
| 11 | flat | 0.00 | 1 | 0.392 | 0.355 | 0.254 | 0.392 | 0.339 | 0.007 | 8.218 | -0.038 | `phase_d4_context_meta_repeat_20260719_165843_d6_mlp_gated_robustness_n5_summary_v0.json` |
| 12 | flat | 0.30 | 1 | 0.391 | 0.354 | 0.255 | 0.391 | 0.432 | 0.000 | 8.250 | -0.316 | `phase_d4_context_meta_repeat_20260719_185241_d6_mlp_gated_yoffset_p030_n3_summary_v0.json` |

## Lowest stability score samples

| rank | context | reset_y | trial | value | beta_m | beta_s | beta_e | max_abs_y | hold_drift | final_x | final_y | source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | goal_flat | -0.60 | 1 | 0.512 | 0.173 | 0.599 | 0.228 | 0.718 | 0.356 | 7.725 | 0.231 | `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json` |
| 2 | downslope | -0.60 | 1 | 0.520 | 0.162 | 0.628 | 0.210 | 0.718 | 0.356 | 7.725 | 0.231 | `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json` |
| 3 | goal_flat | -0.30 | 3 | 0.524 | 0.164 | 0.613 | 0.223 | 1.036 | 0.076 | 8.000 | -1.036 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 4 | downslope | -0.30 | 3 | 0.574 | 0.157 | 0.632 | 0.211 | 1.036 | 0.076 | 8.000 | -1.036 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 5 | rough | -0.60 | 1 | 0.615 | 0.166 | 0.558 | 0.276 | 0.718 | 0.356 | 7.725 | 0.231 | `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json` |
| 6 | rough | -0.30 | 3 | 0.626 | 0.159 | 0.564 | 0.277 | 1.036 | 0.076 | 8.000 | -1.036 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 7 | upslope | -0.60 | 1 | 0.631 | 0.238 | 0.466 | 0.296 | 0.718 | 0.356 | 7.725 | 0.231 | `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json` |
| 8 | flat | -0.60 | 1 | 0.654 | 0.309 | 0.358 | 0.333 | 0.718 | 0.356 | 7.725 | 0.231 | `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json` |
| 9 | goal_flat | -0.30 | 1 | 0.672 | 0.178 | 0.578 | 0.244 | 0.820 | 0.000 | 8.113 | -0.820 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |
| 10 | downslope | 0.30 | 3 | 0.675 | 0.166 | 0.610 | 0.224 | 0.689 | 0.061 | 7.989 | -0.675 | `phase_d4_context_meta_repeat_20260719_185241_d6_mlp_gated_yoffset_p030_n3_summary_v0.json` |
| 11 | goal_flat | 0.30 | 3 | 0.679 | 0.181 | 0.571 | 0.248 | 0.689 | 0.061 | 7.989 | -0.675 | `phase_d4_context_meta_repeat_20260719_185241_d6_mlp_gated_yoffset_p030_n3_summary_v0.json` |
| 12 | downslope | -0.30 | 1 | 0.689 | 0.167 | 0.608 | 0.225 | 0.820 | 0.000 | 8.113 | -0.820 | `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json` |

## Diagnosis

- The bootstrap labels are normalized and usable as first beta targets.
- Energy beta can become high even on successful fast motion because the current energy proxy directly penalizes vx/clearance/yaw effort.
- For D7.0c training, a refined target should keep the semantic terrain prior while increasing stability more strongly for high lateral deviation or hold drift.
- This is still a bootstrap objective selector target, not learned IRL yet.

