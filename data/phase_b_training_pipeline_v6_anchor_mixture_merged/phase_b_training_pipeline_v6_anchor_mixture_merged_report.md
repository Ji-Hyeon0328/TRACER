# Phase-B v6 Anchor-Mixture Merged Dataset

## Sources

- `data/phase_b_training_pipeline_v5_merged_balanced`
- `data/phase_b_anchor_mixture_v8_eval`
- `data/phase_b_anchor_mixture_v8b_sweep_eval`
- `data/phase_b_anchor_mixture_v8c_phase_scheduled_eval`

## Files

| file | rows |
|---|---:|
| `phase_b_rollout_index_v1.jsonl` | 258 |
| `phase_b_objective_preference_pairs_v1.jsonl` | 2686 |
| `phase_b_ram_teacher_student_episode_labels_v1.jsonl` | 258 |
| `phase_b_discrete_theta_lite_ppo_rows_v1.jsonl` | 258 |

## Rollouts by world::action

| world::action | n |
|---|---:|
| earth::trot_cautious | 5 |
| earth::trot_mid | 12 |
| earth::trot_soft_mid_clear | 5 |
| earth::trot_solid_fast | 14 |
| stairs_single::trot_cautious | 5 |
| stairs_single::trot_mid | 10 |
| stairs_single::trot_soft_mid_clear | 5 |
| stairs_single::trot_solid_fast | 16 |
| tracer_sponge_firm_flat::sponge_mid_brake_clear | 5 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 19 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 22 |
| tracer_sponge_firm_flat::sponge_short_step_stable | 5 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 |
| tracer_sponge_firm_flat::sponge_v8_anchor_mix_top3 | 10 |
| tracer_sponge_firm_flat::sponge_v8b_probe_reach_fast | 5 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 5 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias_fast | 10 |
| tracer_sponge_firm_flat::sponge_v8b_reach_stabilized | 5 |
| tracer_sponge_firm_flat::sponge_v8b_soft_reach | 5 |
| tracer_sponge_firm_flat::sponge_v8c_far_fast_early_stop | 5 |
| tracer_sponge_firm_flat::sponge_v8c_mid_fast_early_stop | 5 |
| tracer_sponge_firm_flat::sponge_v8c_soft_stop | 5 |
| tracer_sponge_firm_flat::trot_cautious | 16 |
| tracer_sponge_firm_flat::trot_mid | 17 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 32 |
| tracer_sponge_firm_flat::trot_solid_fast | 10 |
