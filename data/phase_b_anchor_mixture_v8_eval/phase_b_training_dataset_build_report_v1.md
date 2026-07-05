# Phase-B Training Dataset Build Report

- Result root: `artifacts/phase_b_anchor_mixture_v8_eval`
- Output dir: `data/phase_b_anchor_mixture_v8_eval`
- Rollouts: `20`
- Objective preference pairs: `150`
- RAM label rows: `20`
- PPO rows: `20`

| world::action | n | reach_rate | mean_score | mean_risk |
|---|---:|---:|---:|---:|
| tracer_sponge_firm_flat::sponge_probe_crawlish | 5 | 0.200 | 0.317 | 0.860 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 5 | 0.600 | 2.657 | 0.640 |
| tracer_sponge_firm_flat::sponge_v8_anchor_mix_top3 | 5 | 0.000 | -0.556 | 0.900 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 5 | 0.200 | 0.809 | 0.800 |

## Dataset roles

- `phase_b_objective_preference_pairs_v1.jsonl`: bootstrap preference/self-IRL Objective Selector seed pairs.
- `phase_b_ram_teacher_student_episode_labels_v1.jsonl`: RAM teacher-student seed labels from rollout outcomes.
- `phase_b_discrete_theta_lite_ppo_rows_v1.jsonl`: discrete theta-lite PPO episode-level seed rows.

This is a dataset seed pipeline. It does not replace later window-level RAM training or β/RAM-conditioned PPO, but it closes the collection-to-training-data loop.
