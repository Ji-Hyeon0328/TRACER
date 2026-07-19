# TRACER Phase-D6 Milestone Summary v0

## Scope

Phase-D6 extends Phase-D5 from a ridge learned selector to a small MLP learned selector.

The final validated structure is:

```text
terrain context + beta proxy + RAM proxy + odom
  -> MLP learned selector
  -> MLP high-level reference

empirical policy artifact
  -> empirical safety reference

dimension-selective hold-aware gate:
  moving phase:
    use MLP vx
    use MLP yaw_rate
    use MLP clearance
    keep empirical body_h
    keep empirical enable

  goal/hold phase:
    fallback to empirical hold reference

gate output:
  /tracer/mpc_reference
  -> A1-QP-MPC low-level controller
Important commits
5b9e797 Train Phase-D6 MLP selector from Phase-D5 shadow data
4769997 Validate Phase-D6 MLP selector runtime shadow
c504001 Validate Phase-D6 MLP-gated selector control
Important artifacts
D6.0 MLP offline selector
models/phase_d6/d6_mlp_selector_from_d5_shadow_dataset_20260718_215059_v0.pt
models/phase_d6/d6_mlp_selector_from_d5_shadow_dataset_20260718_215059_v0.json
reports/phase_d6/d6_mlp_selector_from_d5_shadow_dataset_20260718_215059_v0.md
reports/phase_d6/d6_ridge_vs_mlp_selector_comparison_v0.md
D6.1 MLP runtime shadow
reports/phase_d6/phase_d4_context_meta_repeat_20260719_143609_d6_mlp_shadow_clamped_final_n3_summary_v0.md
D6.2 MLP-gated control
reports/phase_d4_context_meta_repeat_20260719_163540_manifest.tsv
reports/phase_d6/phase_d4_context_meta_repeat_20260719_163540_d6_mlp_gated_control_final_n3_summary_v0.md
reports/phase_d6/phase_d4_context_meta_repeat_20260719_163540_d6_mlp_gated_control_final_n3_rollouts_v0.csv
Final D6.2 validation

World:

tracer_mixed_stress_course_v5_lowfric_from_solid

Configuration:

goal_x = 8.0
lateral_bound = 2.0
hold_vx = 0.025
hold_obs_s = 60

MLP gated selected dimensions:
  vx=True
  yaw=True
  body_h=False
  clearance=True
  enable=False

Final N=3 result:

success_rate        = 1.000
goal_rate           = 1.000
startup_failed_rate = 0.000
out_lane_rate       = 0.000
hold_drift_mean     = 0.0043333333333333

Individual rollouts:

trial 1:
  final_x=8.169
  final_y=-0.488
  max_abs_y=0.488
  mean_abs_y=0.192
  hold_drift=0.013

trial 2:
  final_x=8.212
  final_y=-0.085
  max_abs_y=0.417
  mean_abs_y=0.201
  hold_drift=0.000

trial 3:
  final_x=8.209
  final_y=0.061
  max_abs_y=0.249
  mean_abs_y=0.093
  hold_drift=0.000

Gate behavior:

trial 1:
  moving_accept_rate = 1774/1784 = 0.994395
  hold_accept_rate   = 0/637 = 0.000000

trial 2:
  moving_accept_rate = 1779/1789 = 0.994410
  hold_accept_rate   = 0/632 = 0.000000

trial 3:
  moving_accept_rate = 1788/1797 = 0.994992
  hold_accept_rate   = 0/623 = 0.000000

Topic wiring:

empirical_ref_topic = /tracer/empirical_mpc_reference
learned_ref_topic   = /tracer/mlp_selector_shadow_ref
gate output topic   = /tracer/mpc_reference
Interpretation

Phase-D6 validates that an MLP learned selector can be placed into the real Gazebo/A1-QP-MPC control loop through a dimension-selective safety gate.

The MLP controls moving-phase vx, yaw_rate, and clearance, while body height, enable, and goal/hold behavior remain protected by the empirical safety reference.

This is still a supervised imitation selector, not the final RL meta-planner. However, it closes the neural learned-selector runtime path and provides a safe interface for later RL or teacher-student extensions.

Next phase

Recommended next step:

D6.3: robustness validation
  - repeat D6.2 under multiple random seeds
  - add initial lateral offsets
  - add small goal/lane variations
  - compare empirical, ridge-gated, and MLP-gated control

D7:
  - improve Objective Selector beta bootstrap
  - build richer RAM teacher-student dataset
  - transition from supervised selector to RL meta-planner

