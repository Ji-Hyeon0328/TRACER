# Phase-B Learned Module Shortlist

This shortlist filters the broad inventory down to likely Objective Selector / RAM candidates for Phase-B integration.

## Summary

| group | exists | connectability | path | size |
|---|---:|---|---|---:|
| objective_selector | True | needs_architecture_check | `artifacts/objective_selector_v1/objective_selector_v1.pt` | 27557 |
| objective_selector | True | needs_architecture_check | `artifacts/objective_selector_v2/objective_selector_v2.pt` | 28069 |
| objective_selector | True | needs_architecture_check | `artifacts/objective_preference_v2/objective_preference_v2.pt` | 27141 |
| objective_selector | True | needs_architecture_check | `artifacts/phase_a_objective_selector_pref_v0/phase_a_objective_selector_pref_v0.pt` | 116103 |
| objective_selector | True | high_priority_objective_candidate | `artifacts/phase_a_objective_selector_pref_with_risk_v0/phase_a_objective_selector_pref_v0.pt` | 116103 |
| objective_selector | True | needs_architecture_check | `artifacts/tracer_objective_irl_v0/model.pt` | 33573 |
| objective_selector | True | high_priority_objective_candidate | `artifacts/tracer_objective_irl_v1/model.pt` | 68389 |
| objective_selector | True | theta_selection_candidate | `artifacts/tracer_highlevel_selector_v0/model.pt` | 71131 |
| objective_selector | True | needs_architecture_check | `artifacts/preference_reward_slide_v0/model.pt` | 24061 |
| objective_selector | True | needs_architecture_check | `artifacts/preference_reward_v0/model.pt` | 23229 |
| objective_selector | True | runtime_export_candidate | `configs/learned_models/tracer_preference_objective_irl_v0_model.json` | 8277 |
| objective_selector | True | runtime_export_candidate | `configs/learned_models/tracer_preference_objective_irl_v1_model.json` | 8296 |
| objective_selector | True | runtime_export_candidate | `configs/highlevel_policy/tracer_objective_conditioned_selector_v0.json` | 12125 |
| ram | True | high_priority_phase_b_candidate | `artifacts/tracer_ram_scalar_v3/model.pt` | 90405 |
| ram | True | high_priority_phase_b_candidate | `configs/learned_models/tracer_ram_scalar_v3_model.pt` | 91381 |
| ram | True | needs_architecture_check | `artifacts/tracer_ram_v0/model.pt` | 218887 |
| ram | True | needs_input_schema_check | `artifacts/tracer_ram_v2/model.pt` | 117925 |
| ram | True | needs_input_schema_check | `artifacts/tracer_ram_v2c/model.pt` | 117925 |
| ram | True | needs_input_schema_check | `artifacts/phase_a_ram_empirical_v1/phase_a_ram_empirical_v1.pt` | 110971 |
| ram | True | needs_input_schema_check | `artifacts/phase_a_ram_teacher_smoke_v0/phase_a_ram_teacher_smoke_v0.pt` | 72691 |
| ram | True | needs_architecture_check | `artifacts/ram_shadow_v2/ram_shadow_v2.pt` | 873957 |
| ram | True | needs_architecture_check | `artifacts/ram_intervention_v1/ram_intervention_v1.pt` | 939077 |
| ram | True | phase_b_proxy_candidate | `configs/phase_b_objective_ram_bootstrap_v0/current_model.json` | 40466 |
| ram | True | phase_b_proxy_candidate | `configs/phase_b_objective_ram_uncertainty_v0/current_registry.json` | 2469 |
| runtime_scripts | True | integration_or_diagnostic_script | `scripts/runtime/check_objective_selector_runtime_v0.py` | 5232 |
| runtime_scripts | True | integration_or_diagnostic_script | `scripts/runtime/tracer_ensure_objective_selector_beta_node_v0.sh` | 1622 |
| runtime_scripts | True | integration_or_diagnostic_script | `scripts/runtime/tracer_publish_objective_beta_once.sh` | 1137 |
| runtime_scripts | True | integration_or_diagnostic_script | `scripts/runtime/tracer_run_learned_stack_v3_beta_blend_robust30.sh` | 1362 |
| runtime_scripts | True | integration_or_diagnostic_script | `scripts/runtime/tracer_run_objective_selector_clean_routing_smoke.sh` | 4011 |
| runtime_scripts | True | integration_or_diagnostic_script | `scripts/runtime/tracer_run_objective_selector_overlay_smoke.sh` | 3807 |
| runtime_scripts | True | integration_or_diagnostic_script | `scripts/training/tracer_check_objective_selector_v1.py` | 1870 |
| runtime_scripts | True | integration_or_diagnostic_script | `scripts/training/tracer_check_objective_conditioned_selector_v0.py` | 1268 |
| runtime_scripts | True | integration_or_diagnostic_script | `scripts/training/tracer_check_preference_objective_irl_v0.py` | 4043 |
| runtime_scripts | True | integration_or_diagnostic_script | `scripts/training/tracer_report_learned_stack_v3_beta_blend.py` | 6683 |
| runtime_scripts | True | integration_or_diagnostic_script | `scripts/training/tracer_report_learned_stack_v3_beta_blend_robust_eval.py` | 5931 |
| runtime_scripts | True | integration_or_diagnostic_script | `scripts/training/tracer_report_learned_stack_v3_beta_shadow.py` | 4735 |
| datasets_reports | True | supporting_data_or_report | `data/preference_datasets/tracer_objective_selector_runtime_v0.jsonl` | 55272 |
| datasets_reports | False | supporting_data_or_report | `data/preference_datasets/tracer_objective_selector_data_v1_core3_hard3_objective_conditioned_pairs_clean.jsonl` |  |
| datasets_reports | True | supporting_data_or_report | `data/training/tracer_highlevel_selector_dataset_v0.csv` | 56718 |
| datasets_reports | True | supporting_data_or_report | `data/training/tracer_irl_trajectory_features_v0.csv` | 112827 |
| datasets_reports | True | supporting_data_or_report | `reports/learned_stack_v3_beta_blend_report.md` | 1021 |
| datasets_reports | True | supporting_data_or_report | `reports/learned_stack_v3_beta_blend_robust_eval_report.md` | 921 |
| datasets_reports | True | supporting_data_or_report | `reports/learned_stack_v3_beta_shadow_report.md` | 950 |
| datasets_reports | True | supporting_data_or_report | `reports/tracer_preference_objective_irl_v1_summary.json` | 4815 |

## Recommended Integration Priority

1. Inspect `tracer_ram_scalar_v3` input/output schema.
2. Inspect `tracer_objective_irl_v1` and `phase_a_objective_selector_pref_with_risk_v0` schema.
3. Check whether existing runtime scripts already export β or selector decisions.
4. Build a Phase-B adapter that maps current episode/window features into those learned modules.
5. Compare raw PPO vs learned Objective Selector/RAM adapter, using reach, stability, speed, and energy proxy.
