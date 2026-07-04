# Phase-B Training Dataset Build Report

- Result root: `artifacts/phase_b_training_campaign_v5_balanced_explore`
- Output dir: `data/phase_b_training_pipeline_v5_balanced_explore`
- Rollouts: `50`
- Objective preference pairs: `403`
- RAM label rows: `50`
- PPO rows: `50`

| world::action | n | reach_rate | mean_score | mean_risk |
|---|---:|---:|---:|---:|
| earth::trot_mid | 5 | 1.000 | 7.929 | 0.000 |
| earth::trot_solid_fast | 5 | 1.000 | 7.917 | 0.000 |
| stairs_single::trot_mid | 5 | 0.600 | 5.648 | 0.280 |
| stairs_single::trot_solid_fast | 5 | 0.800 | 7.096 | 0.080 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 6 | 0.500 | 2.267 | 0.725 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 6 | 0.667 | 3.029 | 0.633 |
| tracer_sponge_firm_flat::trot_cautious | 6 | 0.500 | 1.943 | 0.650 |
| tracer_sponge_firm_flat::trot_mid | 6 | 0.167 | -0.639 | 0.833 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 6 | 0.833 | 3.740 | 0.542 |

## Dataset roles

- `phase_b_objective_preference_pairs_v1.jsonl`: bootstrap preference/self-IRL Objective Selector seed pairs.
- `phase_b_ram_teacher_student_episode_labels_v1.jsonl`: RAM teacher-student seed labels from rollout outcomes.
- `phase_b_discrete_theta_lite_ppo_rows_v1.jsonl`: discrete theta-lite PPO episode-level seed rows.

This is a dataset seed pipeline. It does not replace later window-level RAM training or β/RAM-conditioned PPO, but it closes the collection-to-training-data loop.
