# TRACER Phase-H/I Closure Summary v0

This closes the first Objective Selector + RAM supervised/shadow loop before moving to Phase-J meta-plan learning.

## Phase-H Objective Selector

### H0
Built a rollout-level supervised Objective Selector pretraining table from robust true-metric teacher seeds.

- rows: 22
- feature columns: 57
- deployable-pretrain features: 54
- diagnostic-only features: 3
- target beta source: robust true-metric teacher seeds
- safe claim: supervised pretraining seeds, not final IRL labels

### H1
Trained ridge Objective Selector baselines.

- deployable_pretrain mean LOTO L1: 0.323784
- diagnostic_with_reset_y mean LOTO L1: 0.326379
- worst heldout: p060
- interpretation: reset-y diagnostic features did not meaningfully improve separability

### H2
Added runtime Objective Selector shadow node.

- output topic: /tracer/objective_beta_h2_shadow
- active smoke succeeded with runtime stream
- observed features were mostly available
- H2 remained shadow-only

### H3
Compared H2 beta against existing D7 beta.

- H2 was much more motion-heavy than D7
- actual_beta mean L1 vs D7: 0.740735
- H2 beta mean: approximately (0.614, 0.192, 0.193)
- D7 actual beta mean: approximately (0.302, 0.463, 0.236)
- conclusion: H2 should not be used for active control

### H4
Fed I5 safe RAM into H2 and compared against D7.

- H2 with I5 safe RAM became even more motion-heavy
- actual_beta mean L1 vs D7: 0.930234
- H2 beta mean: approximately (0.757, 0.123, 0.120)
- D7 actual beta mean: approximately (0.303, 0.462, 0.235)
- rough and downslope were especially unsafe:
  - rough H2 ≈ (0.956, 0.002, 0.043), D7 ≈ (0.250, 0.550, 0.200)
  - downslope H2 ≈ (0.909, 0.088, 0.003), D7 ≈ (0.250, 0.600, 0.150)
- conclusion: H2/H4 Objective Selector remains shadow-only

## Phase-I RAM

### I0
Built heuristic RAM teacher dataset.

- rows: 22
- input features: 23
- targets:
  - rho_slip_target
  - rho_rough_target
  - sigma_target
- safe claim: heuristic supervised teacher seeds, not final teacher-student RAM

### I1
Trained RAM ridge predictor.

- mean baseline LOTO L1: 0.601317
- ridge LOTO L1: 0.401097
- worst heldout: p045
- interpretation: heuristic RAM teacher signals are partially predictable from runtime-style features

### I2
Added learned RAM shadow node.

- output topic: /tracer/ram_mismatch_i2_shadow
- active smoke succeeded
- learned RAM was very sharp/saturated
- conclusion: I2 should not be used directly as active RAM

### I3
Compared I2 learned RAM against legacy RAM by context.

- learned RAM saturation was high
- known-context saturation:
  - slip: about 0.90
  - rough: about 0.84
  - sigma: about 0.45
- conclusion: ridge+clamp learned RAM is too sharp for active use

### I4
Built offline damped/safe RAM postprocess.

- learned L1 vs legacy: 0.963292
- damped L1 vs legacy: 0.913200
- safe L1 vs legacy: 0.319620
- learned saturation: high
- damped/safe saturation: zero
- conclusion: safe postprocess is useful

### I5
Added runtime safe RAM postprocess node.

- input: /tracer/ram_mismatch_i2_shadow
- legacy input: /tracer/ram_mismatch
- output: /tracer/ram_mismatch_i5_safe_shadow
- active smoke succeeded
- learned L1 vs legacy: about 1.007
- safe L1 vs legacy: about 0.319
- safe saturation: zero
- conclusion: I5 is a useful shadow RAM candidate, but should not replace legacy RAM yet

## Final H/I decision

For active control, keep:

- D7 Objective Selector
- existing legacy/proxy /tracer/ram_mismatch

For shadow/diagnostic use, keep:

- H2/H4 Objective Selector candidates
- I2 learned RAM
- I5 safe RAM

Do not use H2/H4 as active beta source yet because they are too motion-heavy, especially on rough and downslope terrain.

## Next phase

Move to Phase-J: meta-plan learning / RL preparation.

Recommended J policy:

- Use D7 beta as the active objective signal for now.
- Keep I5 safe RAM as shadow/auxiliary feature.
- Start with a discrete meta-action bank and offline/surrogate evaluation before full online RL.
- Treat the first J-stage as contextual bandit / action-bank policy learning, not final PPO-style RL.
