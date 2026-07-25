# TRACER Phase-L6F Rough-Clearance Final Effect v0

- source: `datasets/phase_l/l6e_fresh_rough_clearance_rollout_evidence_v0.csv`
- reliability uses all six paired repeats.
- continuous quality uses only repeats where both modes completed the rough segment.
- positive signed effect means candidate improvement.
- repeatable classification requires a non-crossing bootstrap CI and at least a 5/6 same-sign proportion.
- candidate-vs-noop is the primary causal comparison.
- candidate-vs-baseline is a secondary sensitivity comparison.

## routing_placebo: rough_noop vs baseline

### Reliability — all six repeats

- `rough_completed`: candidate 5/6, control 6/6; wins/losses/ties 0/1/5; `control_directional_edge`; exact sign p=1
- `goal_reached`: candidate 5/6, control 6/6; wins/losses/ties 0/1/5; `control_directional_edge`; exact sign p=1

### Conditional rough quality — both modes completed

- included repeats: 1, 2, 4, 5, 6
- comparison conclusion: `unresolved`

- `rough_elapsed_span_s`: mean -1.25999322; CI [-3.25995116, 0.300105429]; +/-/tie 2/3/0; `unresolved`; sign p=1
- `rough_progress_rate`: mean -0.00135261615; CI [-0.00354767818, 0.000340435429]; +/-/tie 2/3/0; `unresolved`; sign p=1
- `rough_mean_abs_y`: mean 0.116728405; CI [-0.0146244171, 0.248081228]; +/-/tie 4/1/0; `unresolved`; sign p=0.375
- `rough_max_abs_y`: mean 0.14492517; CI [-0.00923104792, 0.299081387]; +/-/tie 4/1/0; `unresolved`; sign p=0.375
- `rough_end_abs_y`: mean 0.101491847; CI [-0.10628227, 0.309265964]; +/-/tie 3/2/0; `unresolved`; sign p=1
- `rough_lateral_growth`: mean -0.0230337357; CI [-0.141150966, 0.121905052]; +/-/tie 2/3/0; `unresolved`; sign p=1

## candidate_vs_noop: rough_clear_low05 vs rough_noop

### Reliability — all six repeats

- `rough_completed`: candidate 6/6, control 5/6; wins/losses/ties 1/0/5; `candidate_directional_edge`; exact sign p=1
- `goal_reached`: candidate 5/6, control 5/6; wins/losses/ties 1/1/4; `no_directional_edge`; exact sign p=1

### Conditional rough quality — both modes completed

- included repeats: 1, 2, 4, 5, 6
- comparison conclusion: `unresolved`

- `rough_elapsed_span_s`: mean -12.6200882; CI [-39.6603726, 1.84002714]; +/-/tie 3/2/0; `unresolved`; sign p=1
- `rough_progress_rate`: mean -0.00494767155; CI [-0.0167080164, 0.0019237994]; +/-/tie 3/2/0; `unresolved`; sign p=1
- `rough_mean_abs_y`: mean -0.0257977986; CI [-0.153997983, 0.0625505471]; +/-/tie 3/2/0; `unresolved`; sign p=1
- `rough_max_abs_y`: mean -0.0339874; CI [-0.154917, 0.0720264]; +/-/tie 2/3/0; `unresolved`; sign p=1
- `rough_end_abs_y`: mean 0.0129146; CI [-0.17255, 0.1837492]; +/-/tie 3/2/0; `unresolved`; sign p=1
- `rough_lateral_growth`: mean 0.0762564; CI [-0.010863, 0.168543]; +/-/tie 3/2/0; `unresolved`; sign p=1

## candidate_vs_baseline: rough_clear_low05 vs baseline

### Reliability — all six repeats

- `rough_completed`: candidate 6/6, control 6/6; wins/losses/ties 0/0/6; `no_directional_edge`; exact sign p=1
- `goal_reached`: candidate 5/6, control 6/6; wins/losses/ties 0/1/5; `control_directional_edge`; exact sign p=1

### Conditional rough quality — both modes completed

- included repeats: 1, 2, 3, 4, 5, 6
- comparison conclusion: `unresolved`

- `rough_elapsed_span_s`: mean -11.6000031; CI [-33.8498493, 0.433236957]; +/-/tie 2/4/0; `unresolved`; sign p=0.6875
- `rough_progress_rate`: mean -0.00529206162; CI [-0.0147489673, 0.000457334556]; +/-/tie 2/4/0; `unresolved`; sign p=0.6875
- `rough_mean_abs_y`: mean 0.129113962; CI [-0.0526682126, 0.28739728]; +/-/tie 4/2/0; `unresolved`; sign p=0.6875
- `rough_max_abs_y`: mean 0.151338681; CI [-0.0432553349, 0.345812004]; +/-/tie 4/2/0; `unresolved`; sign p=0.6875
- `rough_end_abs_y`: mean 0.139495174; CI [-0.0643470545, 0.349464131]; +/-/tie 4/2/0; `unresolved`; sign p=0.6875
- `rough_lateral_growth`: mean 0.0525085945; CI [-0.0438593945, 0.146094208]; +/-/tie 4/2/0; `unresolved`; sign p=0.6875

## candidate_vs_noop: rough_clear_high05 vs rough_noop

### Reliability — all six repeats

- `rough_completed`: candidate 6/6, control 5/6; wins/losses/ties 1/0/5; `candidate_directional_edge`; exact sign p=1
- `goal_reached`: candidate 6/6, control 5/6; wins/losses/ties 1/0/5; `candidate_directional_edge`; exact sign p=1

### Conditional rough quality — both modes completed

- included repeats: 1, 2, 4, 5, 6
- comparison conclusion: `progress_cost_with_lateral_degradation_signal`

- `rough_elapsed_span_s`: mean -1.43999276; CI [-2.45983686, -0.660227013]; +/-/tie 0/5/0; `repeatable_negative`; sign p=0.0625
- `rough_progress_rate`: mean -0.00149076161; CI [-0.00253300547, -0.000713853158]; +/-/tie 0/5/0; `repeatable_negative`; sign p=0.0625
- `rough_mean_abs_y`: mean -0.0880300822; CI [-0.158264169, -0.0211544456]; +/-/tie 1/4/0; `negative_ci_sign_shortfall`; sign p=0.375
- `rough_max_abs_y`: mean -0.1013674; CI [-0.1952044, -0.0201224]; +/-/tie 1/4/0; `negative_ci_sign_shortfall`; sign p=0.375
- `rough_end_abs_y`: mean -0.05697; CI [-0.2126702, 0.0820642]; +/-/tie 2/3/0; `unresolved`; sign p=1
- `rough_lateral_growth`: mean 0.0056028; CI [-0.120311, 0.136457]; +/-/tie 3/2/0; `unresolved`; sign p=1

## candidate_vs_baseline: rough_clear_high05 vs baseline

### Reliability — all six repeats

- `rough_completed`: candidate 6/6, control 6/6; wins/losses/ties 0/0/6; `no_directional_edge`; exact sign p=1
- `goal_reached`: candidate 6/6, control 6/6; wins/losses/ties 0/0/6; `no_directional_edge`; exact sign p=1

### Conditional rough quality — both modes completed

- included repeats: 1, 2, 3, 4, 5, 6
- comparison conclusion: `repeatable_progress_cost`

- `rough_elapsed_span_s`: mean -2.46639637; CI [-3.8664966, -1.29981287]; +/-/tie 0/6/0; `repeatable_negative`; sign p=0.03125
- `rough_progress_rate`: mean -0.00259252736; CI [-0.00406912885, -0.00131398594]; +/-/tie 0/6/0; `repeatable_negative`; sign p=0.03125
- `rough_mean_abs_y`: mean 0.0232442033; CI [-0.0887716339, 0.149044351]; +/-/tie 3/3/0; `unresolved`; sign p=1
- `rough_max_abs_y`: mean 0.0335385146; CI [-0.0899594945, 0.193073106]; +/-/tie 3/3/0; `unresolved`; sign p=1
- `rough_end_abs_y`: mean 0.0454038404; CI [-0.106439469, 0.267915433]; +/-/tie 3/3/0; `unresolved`; sign p=1
- `rough_lateral_growth`: mean 0.0237737611; CI [-0.128529528, 0.176932473]; +/-/tie 4/2/0; `unresolved`; sign p=0.6875

## Safety and interpretation

- Reliability edges with only one discordant pair are descriptive, not statistical confirmation.
- Conditional quality excludes the rough-incomplete no-op repeat, while its failure remains in reliability analysis.
- The low-clearance repeat that completed rough slowly and later missed the goal remains in both reliability and conditional rough-quality analysis.
- No action label is promoted unless the primary candidate-vs-noop comparison shows a repeatable useful trade-off.
- Empirical remains the active default.
