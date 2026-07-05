# Phase-B Training Dataset Build Report

- Result root: `artifacts/phase_b_anchor_mixture_v8b_sweep_eval`
- Output dir: `data/phase_b_anchor_mixture_v8b_sweep_eval`
- Rollouts: `40`
- Objective preference pairs: `627`
- RAM label rows: `40`
- PPO rows: `40`

| world::action | n | reach_rate | mean_score | mean_risk |
|---|---:|---:|---:|---:|
| tracer_sponge_firm_flat::sponge_reach_then_brake | 5 | 0.600 | 2.521 | 0.640 |
| tracer_sponge_firm_flat::sponge_v8_anchor_mix_top3 | 5 | 0.800 | 3.453 | 0.560 |
| tracer_sponge_firm_flat::sponge_v8b_probe_reach_fast | 5 | 0.800 | 3.690 | 0.530 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 5 | 0.600 | 2.945 | 0.670 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias_fast | 5 | 1.000 | 4.997 | 0.450 |
| tracer_sponge_firm_flat::sponge_v8b_reach_stabilized | 5 | 0.800 | 3.612 | 0.530 |
| tracer_sponge_firm_flat::sponge_v8b_soft_reach | 5 | 0.400 | 1.266 | 0.750 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 5 | 0.600 | 2.695 | 0.600 |

## Dataset roles

- `phase_b_objective_preference_pairs_v1.jsonl`: bootstrap preference/self-IRL Objective Selector seed pairs.
- `phase_b_ram_teacher_student_episode_labels_v1.jsonl`: RAM teacher-student seed labels from rollout outcomes.
- `phase_b_discrete_theta_lite_ppo_rows_v1.jsonl`: discrete theta-lite PPO episode-level seed rows.

This is a dataset seed pipeline. It does not replace later window-level RAM training or β/RAM-conditioned PPO, but it closes the collection-to-training-data loop.
