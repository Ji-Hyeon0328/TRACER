# Phase-B Training Dataset Build Report

- Result root: `artifacts/phase_b_anchor_mixture_v8c_phase_scheduled_eval`
- Output dir: `data/phase_b_anchor_mixture_v8c_phase_scheduled_eval`
- Rollouts: `25`
- Objective preference pairs: `248`
- RAM label rows: `25`
- PPO rows: `25`

| world::action | n | reach_rate | mean_score | mean_risk |
|---|---:|---:|---:|---:|
| tracer_sponge_firm_flat::sponge_v8b_reach_bias_fast | 5 | 0.000 | -1.278 | 0.910 |
| tracer_sponge_firm_flat::sponge_v8c_far_fast_early_stop | 5 | 0.600 | 2.688 | 0.670 |
| tracer_sponge_firm_flat::sponge_v8c_mid_fast_early_stop | 5 | 0.800 | 2.921 | 0.560 |
| tracer_sponge_firm_flat::sponge_v8c_soft_stop | 5 | 0.600 | 2.400 | 0.670 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 5 | 0.800 | 3.779 | 0.560 |

## Dataset roles

- `phase_b_objective_preference_pairs_v1.jsonl`: bootstrap preference/self-IRL Objective Selector seed pairs.
- `phase_b_ram_teacher_student_episode_labels_v1.jsonl`: RAM teacher-student seed labels from rollout outcomes.
- `phase_b_discrete_theta_lite_ppo_rows_v1.jsonl`: discrete theta-lite PPO episode-level seed rows.

This is a dataset seed pipeline. It does not replace later window-level RAM training or β/RAM-conditioned PPO, but it closes the collection-to-training-data loop.
