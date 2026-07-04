# Phase-B Training Dataset Build Report

- Result root: `artifacts/phase_b_beta_ram_candidate_policy_v3_online_explore_v1`
- Output dir: `data/phase_b_training_pipeline_v4_online_explore`
- Rollouts: `18`
- Objective preference pairs: `30`
- RAM label rows: `18`
- PPO rows: `18`

| world::action | n | reach_rate | mean_score | mean_risk |
|---|---:|---:|---:|---:|
| earth::trot_mid | 2 | 1.000 | 7.919 | 0.000 |
| earth::trot_solid_fast | 4 | 0.500 | 4.061 | 0.388 |
| stairs_single::trot_solid_fast | 6 | 1.000 | 8.269 | 0.017 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 3 | 0.333 | 1.878 | 0.617 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 1 | 1.000 | 4.246 | 0.450 |
| tracer_sponge_firm_flat::trot_mid | 1 | 1.000 | 4.629 | 0.450 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 1 | 1.000 | 8.316 | 0.450 |

## Dataset roles

- `phase_b_objective_preference_pairs_v1.jsonl`: bootstrap preference/self-IRL Objective Selector seed pairs.
- `phase_b_ram_teacher_student_episode_labels_v1.jsonl`: RAM teacher-student seed labels from rollout outcomes.
- `phase_b_discrete_theta_lite_ppo_rows_v1.jsonl`: discrete theta-lite PPO episode-level seed rows.

This is a dataset seed pipeline. It does not replace later window-level RAM training or β/RAM-conditioned PPO, but it closes the collection-to-training-data loop.
