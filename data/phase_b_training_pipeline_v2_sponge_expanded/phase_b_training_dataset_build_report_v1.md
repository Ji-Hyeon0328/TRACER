# Phase-B Training Dataset Build Report

- Result root: `artifacts/phase_b_training_campaign_v2_sponge_expanded`
- Output dir: `data/phase_b_training_pipeline_v2_sponge_expanded`
- Rollouts: `45`
- Objective preference pairs: `791`
- RAM label rows: `45`
- PPO rows: `45`

| world::action | n | reach_rate | mean_score | mean_risk |
|---|---:|---:|---:|---:|
| tracer_sponge_firm_flat::sponge_mid_brake_clear | 5 | 0.400 | 1.539 | 0.720 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 5 | 0.800 | 3.825 | 0.560 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 5 | 0.800 | 3.475 | 0.530 |
| tracer_sponge_firm_flat::sponge_short_step_stable | 5 | 0.400 | 1.367 | 0.750 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.200 | 0.446 | 0.860 |
| tracer_sponge_firm_flat::trot_cautious | 5 | 0.800 | 3.647 | 0.470 |
| tracer_sponge_firm_flat::trot_mid | 5 | 0.800 | 3.728 | 0.530 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 5 | 0.800 | 3.266 | 0.560 |
| tracer_sponge_firm_flat::trot_solid_fast | 5 | 0.600 | 2.668 | 0.640 |

## Dataset roles

- `phase_b_objective_preference_pairs_v1.jsonl`: bootstrap preference/self-IRL Objective Selector seed pairs.
- `phase_b_ram_teacher_student_episode_labels_v1.jsonl`: RAM teacher-student seed labels from rollout outcomes.
- `phase_b_discrete_theta_lite_ppo_rows_v1.jsonl`: discrete theta-lite PPO episode-level seed rows.

This is a dataset seed pipeline. It does not replace later window-level RAM training or β/RAM-conditioned PPO, but it closes the collection-to-training-data loop.
