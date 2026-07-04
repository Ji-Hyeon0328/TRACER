# Phase-B Learned Module Schema Inspection

This report inspects candidate learned Objective Selector/RAM artifacts for Phase-B integration.

| kind | exists | path | connectability | outputs_possible | normalization | ok |
|---|---:|---|---|---|---:|---:|
| ram | True | `artifacts/tracer_ram_scalar_v3/model.pt` | high_priority_ram_adapter_candidate |  | True | True |
| ram | True | `configs/learned_models/tracer_ram_scalar_v3_model.pt` | high_priority_ram_adapter_candidate |  | True | True |
| ram | True | `artifacts/tracer_ram_v2/model.pt` | needs_manual_schema_match |  | True | True |
| ram | True | `artifacts/tracer_ram_v2c/model.pt` | needs_manual_schema_match |  | True | True |
| ram | True | `artifacts/phase_a_ram_empirical_v1/phase_a_ram_empirical_v1.pt` | needs_manual_schema_match | beta,risk,uncertainty,action | False | True |
| ram | True | `artifacts/phase_a_ram_teacher_smoke_v0/phase_a_ram_teacher_smoke_v0.pt` | needs_manual_schema_match | beta,risk,uncertainty | False | True |
| ram | True | `configs/phase_b_objective_ram_bootstrap_v0/current_model.json` | phase_b_proxy_immediately_readable | beta,risk,preference,action | True | True |
| ram | True | `configs/phase_b_objective_ram_uncertainty_v0/current_registry.json` | phase_b_proxy_immediately_readable | risk,uncertainty,score | True | True |
| objective_selector | True | `artifacts/tracer_objective_irl_v1/model.pt` | high_priority_objective_adapter_candidate | beta | False | True |
| objective_selector | True | `artifacts/tracer_objective_irl_v0/model.pt` | needs_manual_schema_match | beta | False | True |
| objective_selector | True | `artifacts/phase_a_objective_selector_pref_with_risk_v0/phase_a_objective_selector_pref_v0.pt` | risk_aware_objective_candidate | beta,preference,score,action | True | True |
| objective_selector | True | `artifacts/phase_a_objective_selector_pref_v0/phase_a_objective_selector_pref_v0.pt` | needs_manual_schema_match | beta,preference,score,action | True | True |
| objective_selector | True | `artifacts/objective_selector_v2/objective_selector_v2.pt` | needs_manual_schema_match | beta,action | True | True |
| objective_selector | True | `artifacts/objective_selector_v1/objective_selector_v1.pt` | needs_manual_schema_match | beta | True | True |
| objective_selector | True | `artifacts/objective_preference_v2/objective_preference_v2.pt` | needs_manual_schema_match | reward,preference,action | True | True |
| objective_selector | True | `artifacts/tracer_highlevel_selector_v0/model.pt` | needs_manual_schema_match | beta,score | True | True |
| objective_selector | True | `configs/learned_models/tracer_preference_objective_irl_v1_model.json` | runtime_json_candidate | beta,reward,preference,score | True | True |
| objective_selector | True | `configs/learned_models/tracer_preference_objective_irl_v0_model.json` | runtime_json_candidate | beta,reward,preference,score | True | True |
| objective_selector | True | `configs/highlevel_policy/tracer_objective_conditioned_selector_v0.json` | runtime_json_candidate | beta,preference,score | False | True |
| ram | True | `artifacts/tracer_ram_v2/normalization.json` | runtime_json_candidate |  | True | True |
| ram | True | `artifacts/tracer_ram_v0/scaler_metrics_env.json` | runtime_json_candidate | risk,rho,sigma,score | True | True |
| objective_selector | True | `artifacts/tracer_highlevel_selector_v0/scaler_and_metrics.json` | runtime_json_candidate | beta,score | True | True |
| objective_selector | True | `artifacts/tracer_objective_irl_v1/scaler_metrics_env.json` | high_priority_objective_adapter_candidate | beta | True | True |
| objective_selector | True | `artifacts/tracer_objective_irl_v1/objective_beta_by_terrain_v1.csv` | high_priority_objective_adapter_candidate | beta | False | True |
| objective_selector | True | `artifacts/phase_a_objective_selector_pref_with_risk_v0/phase_a_objective_selector_pref_v0_meta.json` | risk_aware_objective_candidate | risk,preference | False | True |
| objective_selector | True | `reports/tracer_preference_objective_irl_v1_summary.json` | runtime_json_candidate | beta,preference,score | True | True |
| objective_selector | True | `reports/learned_stack_v3_beta_blend_report.md` | needs_manual_schema_match | beta | True | True |
| objective_selector | True | `reports/learned_stack_v3_beta_blend_robust_eval_report.md` | needs_manual_schema_match | beta | True | True |

## Priority Interpretation

- `tracer_ram_scalar_v3` should be checked first as a RAM adapter candidate.
- `tracer_objective_irl_v1` and `phase_a_objective_selector_pref_with_risk_v0` should be checked first for Objective Selector / β / risk-aware scoring.
- Phase-B JSON artifacts can be used immediately as proxy/registry inputs, but they are not the same as learned checkpoint inference.
- The next patch should build a small adapter only after confirming input feature names and normalization.
