# Phase-B Action Bank Values v1

This report uses the same `build_action_bank()` parser as the Phase-B campaign runner.

| world(s) | action | found | vx_far | vx_near | slow_dist | stop_dist | body_h | clearance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| earth | trot_mid | True | 0.12 | 0.06 | 0.25 | 0.15 | 0.33 | 0.05 |
| earth | trot_solid_fast | True | 0.16 | 0.08 | 0.2 | 0.15 | 0.32 | 0.04 |
| stairs_single | trot_mid | True | 0.12 | 0.06 | 0.25 | 0.15 | 0.33 | 0.05 |
| stairs_single | trot_solid_fast | True | 0.16 | 0.08 | 0.2 | 0.15 | 0.32 | 0.04 |
| tracer_sponge_firm_flat | sponge_v8b_reach_bias | True | 0.076 | 0.00875 | 0.43999999999999995 | 0.15749999999999997 | 0.3355 | 0.07075000000000001 |
| tracer_sponge_firm_flat | sponge_v1d_stabilized_late_hold_035_016 | True | 0.07600000000000001 | 0.015 | 0.35 | 0.16 | 0.34124999999999994 | 0.08124999999999999 |
| tracer_sponge_firm_flat | sponge_v1d_bias_late_hold_025_013 | True | 0.076 | 0.015 | 0.25 | 0.13 | 0.344925 | 0.08326249999999999 |
| tracer_sponge_firm_flat | sponge_slow_high_clear | True | 0.06 | 0.015 | 0.45 | 0.15 | 0.35 | 0.09 |

## Raw theta entries

### trot_mid

- config: `configs/phase_b_ablation_data_v1/training_campaign_earth_ablation_data_v1.json`
- teacher: `configs/phase_b_training_pipeline_v2_sponge_expanded/ppo_warmstart_teacher_table_sponge_expanded_v0.json`
- worlds: `earth`
- found: `True`

```python
{'body_height': 0.33,
 'goal_slow_distance': 0.25,
 'goal_stop_distance': 0.15,
 'name': 'trot_mid',
 'swing_clearance': 0.05,
 'vx_far': 0.12,
 'vx_near': 0.06}
```

### trot_solid_fast

- config: `configs/phase_b_ablation_data_v1/training_campaign_earth_ablation_data_v1.json`
- teacher: `configs/phase_b_training_pipeline_v2_sponge_expanded/ppo_warmstart_teacher_table_sponge_expanded_v0.json`
- worlds: `earth`
- found: `True`

```python
{'body_height': 0.32,
 'goal_slow_distance': 0.2,
 'goal_stop_distance': 0.15,
 'name': 'trot_solid_fast',
 'swing_clearance': 0.04,
 'vx_far': 0.16,
 'vx_near': 0.08}
```

### trot_mid

- config: `configs/phase_b_ablation_data_v1/training_campaign_stairs_ablation_data_v1.json`
- teacher: `configs/phase_b_training_pipeline_v2_sponge_expanded/ppo_warmstart_teacher_table_sponge_expanded_v0.json`
- worlds: `stairs_single`
- found: `True`

```python
{'body_height': 0.33,
 'goal_slow_distance': 0.25,
 'goal_stop_distance': 0.15,
 'name': 'trot_mid',
 'swing_clearance': 0.05,
 'vx_far': 0.12,
 'vx_near': 0.06}
```

### trot_solid_fast

- config: `configs/phase_b_ablation_data_v1/training_campaign_stairs_ablation_data_v1.json`
- teacher: `configs/phase_b_training_pipeline_v2_sponge_expanded/ppo_warmstart_teacher_table_sponge_expanded_v0.json`
- worlds: `stairs_single`
- found: `True`

```python
{'body_height': 0.32,
 'goal_slow_distance': 0.2,
 'goal_stop_distance': 0.15,
 'name': 'trot_solid_fast',
 'swing_clearance': 0.04,
 'vx_far': 0.16,
 'vx_near': 0.08}
```

### sponge_v8b_reach_bias

- config: `configs/phase_b_ablation_data_v1/training_campaign_sponge_ablation_data_v1.json`
- teacher: `configs/phase_b_sponge_late_switch_v1d/ppo_warmstart_teacher_table_sponge_late_switch_v1d.json`
- worlds: `tracer_sponge_firm_flat`
- found: `True`

```python
{'body_height': 0.3355,
 'goal_slow_distance': 0.43999999999999995,
 'goal_stop_distance': 0.15749999999999997,
 'name': 'sponge_v8b_reach_bias',
 'swing_clearance': 0.07075000000000001,
 'vx_far': 0.076,
 'vx_near': 0.00875}
```

### sponge_v1d_stabilized_late_hold_035_016

- config: `configs/phase_b_ablation_data_v1/training_campaign_sponge_ablation_data_v1.json`
- teacher: `configs/phase_b_sponge_late_switch_v1d/ppo_warmstart_teacher_table_sponge_late_switch_v1d.json`
- worlds: `tracer_sponge_firm_flat`
- found: `True`

```python
{'body_height': 0.34124999999999994,
 'goal_slow_distance': 0.35,
 'goal_stop_distance': 0.16,
 'name': 'sponge_v1d_stabilized_late_hold_035_016',
 'swing_clearance': 0.08124999999999999,
 'vx_far': 0.07600000000000001,
 'vx_near': 0.015}
```

### sponge_v1d_bias_late_hold_025_013

- config: `configs/phase_b_ablation_data_v1/training_campaign_sponge_ablation_data_v1.json`
- teacher: `configs/phase_b_sponge_late_switch_v1d/ppo_warmstart_teacher_table_sponge_late_switch_v1d.json`
- worlds: `tracer_sponge_firm_flat`
- found: `True`

```python
{'body_height': 0.344925,
 'goal_slow_distance': 0.25,
 'goal_stop_distance': 0.13,
 'name': 'sponge_v1d_bias_late_hold_025_013',
 'swing_clearance': 0.08326249999999999,
 'vx_far': 0.076,
 'vx_near': 0.015}
```

### sponge_slow_high_clear

- config: `configs/phase_b_ablation_data_v1/training_campaign_sponge_ablation_data_v1.json`
- teacher: `configs/phase_b_sponge_late_switch_v1d/ppo_warmstart_teacher_table_sponge_late_switch_v1d.json`
- worlds: `tracer_sponge_firm_flat`
- found: `True`

```python
{'body_height': 0.35,
 'goal_slow_distance': 0.45,
 'goal_stop_distance': 0.15,
 'name': 'sponge_slow_high_clear',
 'swing_clearance': 0.09,
 'vx_far': 0.06,
 'vx_near': 0.015}
```

