# TRACER Phase-D5 Milestone Summary v0

## Scope

Phase-D5 closes the high-level learned-selector path on top of the existing A1-QP-MPC Gazebo baseline.

The final validated structure is:

```text
terrain context + beta + RAM proxy + odom
  -> learned selector ridge model
  -> learned high-level reference

empirical policy artifact
  -> empirical safety reference

hold-aware safety gate:
  moving phase: accept learned reference if close to empirical reference
  goal/hold phase: fallback to empirical hold reference

gate output
  -> /tracer/mpc_reference
  -> A1-QP-MPC low-level controller
Important artifacts
D5.1 shadow dataset
datasets/phase_d5/phase_d5_shadow_dataset_20260718_215059_dataset_v0.csv
datasets/phase_d5/phase_d5_shadow_dataset_20260718_215059_summary_v0.md
D5.2 empirical policy artifact
models/phase_d5/d5_empirical_policy_from_shadow_dataset_20260718_215059_v0.json
reports/phase_d5/d5_empirical_policy_from_shadow_dataset_20260718_215059_v0.md
D5.3 learned selector model
models/phase_d5/d5_learned_selector_ridge_from_shadow_dataset_20260718_215059_v0.json
reports/phase_d5/d5_learned_selector_ridge_from_shadow_dataset_20260718_215059_v0.md
reports/phase_d5/d5_learned_selector_ridge_predictions_20260718_215059_v0.md
D5.4 learned selector runtime shadow
reports/phase_d5/phase_d4_context_meta_repeat_20260719_045043_d5_learned_shadow_final_n3_summary_v0.md
D5.5 hold-aware gate dry-run
reports/phase_d5/phase_d4_context_meta_repeat_20260719_055848_d5_gate_dryrun_holdaware_final_n3_summary_v0.md
D5.6 gated learned selector control
reports/phase_d4_context_meta_repeat_20260719_062329_manifest.tsv
reports/phase_d5/phase_d4_context_meta_repeat_20260719_062329_d5_gated_control_final_n3_summary_v0.md
reports/phase_d5/phase_d4_context_meta_repeat_20260719_062329_d5_gated_control_final_n3_rollouts_v0.csv
Final D5.6 validation

World:

tracer_mixed_stress_course_v5_lowfric_from_solid

Goal:

goal_x = 8.0
lateral_bound = 2.0
hold_vx = 0.025
hold_obs_s = 60

Final N=3 result:

success_rate        = 1.000
goal_rate           = 1.000
startup_failed_rate = 0.000
out_lane_rate       = 0.000
hold_drift_mean     = 0.006333333333333376

Individual rollouts:

trial 1: final_x=8.204, final_y=-0.370, max_abs_y=0.370, mean_abs_y=0.170, hold_drift=0.000
trial 2: final_x=8.129, final_y=-0.238, max_abs_y=0.371, mean_abs_y=0.182, hold_drift=0.019
trial 3: final_x=8.219, final_y=-0.191, max_abs_y=0.387, mean_abs_y=0.233, hold_drift=0.000

Gate behavior:

trial 1 moving_accept_rate = 1779/1779 = 1.000000
trial 2 moving_accept_rate = 1757/1759 = 0.998863
trial 3 moving_accept_rate = 1739/1740 = 0.999425

hold_accept_rate = 0.000000 for all trials

Topic wiring confirmed:

empirical_ref_topic = /tracer/empirical_mpc_reference
learned_ref_topic   = /tracer/learned_selector_shadow_ref
gate output topic   = /tracer/mpc_reference
Interpretation

Phase-D5 validates that the learned selector can control the robot during moving terrain segments through a safety gate, while the empirical policy remains responsible for stable goal/hold behavior.

This is not yet a full end-to-end RL policy. It is a supervised learned selector trained from Phase-D5 shadow data and deployed through a hold-aware safety gate.

Next phase

Recommended next phase:

D6.0: replace ridge selector with a small MLP teacher-student selector
D6.1: compare ridge vs MLP in shadow mode
D6.2: deploy MLP through the same hold-aware gate
D6.3: add more diverse worlds / randomized initial lateral offsets

