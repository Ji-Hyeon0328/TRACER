# TRACER Phase-D7.0a Objective Selector Bootstrap Dataset Summary v0

## Purpose

This dataset bootstraps a data-derived Objective Selector from Phase-D6 rollout results.

It is not yet IRL. It converts rollout statistics into interpretable proxy objective scores and heuristic beta targets.

## Dataset

- output_csv: `datasets/phase_d7/d7_objective_bootstrap_dataset_v0.csv`
- samples: `115`
- min_context_rows: `20`

## Context counts

- downslope: `23`
- flat: `23`
- goal_flat: `23`
- rough: `23`
- upslope: `23`

## Target beta / score means

- target_beta_motion_mean: `0.239224`
- target_beta_stability_mean: `0.444608`
- target_beta_energy_mean: `0.316168`
- motion_score_mean: `0.970549`
- stability_score_mean: `0.855155`
- energy_score_mean: `0.556982`

## Source summaries

- `phase_d4_context_meta_repeat_20260719_162757_d6_mlp_gated_control_smoke_n1_summary_v0.json`: `5` context samples
- `phase_d4_context_meta_repeat_20260719_163540_d6_mlp_gated_control_final_n3_summary_v0.json`: `15` context samples
- `phase_d4_context_meta_repeat_20260719_165843_d6_mlp_gated_robustness_n5_summary_v0.json`: `25` context samples
- `phase_d4_context_meta_repeat_20260719_174208_d6_mlp_gated_yoffset_p030_smoke_n1_summary_v0.json`: `5` context samples
- `phase_d4_context_meta_repeat_20260719_180101_d6_mlp_gated_yoffset_p030_smoke_n1_retry_summary_v0.json`: `5` context samples
- `phase_d4_context_meta_repeat_20260719_180917_d6_mlp_gated_yoffset_p030_smoke_n1_verified_summary_v0.json`: `5` context samples
- `phase_d4_context_meta_repeat_20260719_182201_d6_mlp_gated_yoffset_m030_n3_summary_v0.json`: `15` context samples
- `phase_d4_context_meta_repeat_20260719_183716_d6_mlp_gated_yoffset_p000_n3_summary_v0.json`: `15` context samples
- `phase_d4_context_meta_repeat_20260719_185241_d6_mlp_gated_yoffset_p030_n3_summary_v0.json`: `15` context samples
- `phase_d4_context_meta_repeat_20260719_191212_d6_mlp_gated_yoffset_m060_smoke_n1_summary_v0.json`: `5` context samples
- `phase_d4_context_meta_repeat_20260719_191713_d6_mlp_gated_yoffset_p060_smoke_n1_summary_v0.json`: `5` context samples

## Skipped inputs

- none

## Notes

- Input beta/RAM values are still proxy values from the Phase-D5/D6 pipeline.
- Target beta values are bootstrap labels derived from motion, stability, and energy proxy scores.
- Goal/hold behavior remains safety-protected by empirical fallback in the current runtime.
- This dataset is meant to support D7.0b beta-target inspection and D7.0c Objective Selector training.
