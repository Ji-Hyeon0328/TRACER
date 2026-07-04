# Phase-B Training Dataset Build Report

- Result root: `artifacts/phase_b_training_campaign_v1`
- Output dir: `data/phase_b_training_pipeline_v1`
- Rollouts: `60`
- Objective preference pairs: `437`
- RAM label rows: `60`
- PPO rows: `60`

| world::action | n | reach_rate | mean_score | mean_risk |
|---|---:|---:|---:|---:|
| earth::trot_cautious | 5 | 0.200 | 3.711 | 0.320 |
| earth::trot_mid | 5 | 1.000 | 7.926 | 0.000 |
| earth::trot_soft_mid_clear | 5 | 0.000 | 1.871 | 0.550 |
| earth::trot_solid_fast | 5 | 1.000 | 7.910 | 0.000 |
| stairs_single::trot_cautious | 5 | 0.400 | 4.163 | 0.330 |
| stairs_single::trot_mid | 5 | 0.600 | 5.736 | 0.190 |
| stairs_single::trot_soft_mid_clear | 5 | 0.200 | 2.561 | 0.440 |
| stairs_single::trot_solid_fast | 5 | 1.000 | 8.474 | 0.020 |
| tracer_sponge_firm_flat::trot_cautious | 5 | 0.400 | 1.588 | 0.780 |
| tracer_sponge_firm_flat::trot_mid | 5 | 0.600 | 3.348 | 0.580 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 5 | 0.600 | 2.230 | 0.670 |
| tracer_sponge_firm_flat::trot_solid_fast | 5 | 0.200 | -0.436 | 0.830 |

## Dataset roles

- `phase_b_objective_preference_pairs_v1.jsonl`: bootstrap preference/self-IRL Objective Selector seed pairs.
- `phase_b_ram_teacher_student_episode_labels_v1.jsonl`: RAM teacher-student seed labels from rollout outcomes.
- `phase_b_discrete_theta_lite_ppo_rows_v1.jsonl`: discrete theta-lite PPO episode-level seed rows.

This is a dataset seed pipeline. It does not replace later window-level RAM training or β/RAM-conditioned PPO, but it closes the collection-to-training-data loop.
