# LL1 VX A/B Confirmatory Non-Promotion Freeze

## Scope

This record freezes the five-block balanced A/B confirmation for the
LL1 VX terrain schedule in
`tracer_mixed_stress_course_v5_lowfric_from_solid`.

This confirmation follows the earlier three-block A/B/C screening frozen
at base commit `ad60eb53cab323583bc054c95bff30b7d4666997`.

It is not evidence of:

- cross-world generalization,
- battery electrical-energy optimality,
- a final Objective Selector teacher label,
- promotion of the tested B schedule,
- promotion of fixed-high VX on difficult terrain.

## Conditions

### A — fixed low

- all active terrain contexts: `vx=0.090 m/s`

### B — terrain scheduled

- flat/start-flat: `vx=0.115 m/s`
- upslope/rough/downslope: `vx=0.090 m/s`
- unknown: fail closed to `vx=0.090 m/s`

Fixed in both conditions:

- body height: `0.320 m`
- swing clearance: `0.045 m`
- yaw command: `0`
- LL1 frozen A1-QP-MPC runtime

## Balanced paired design

Original blocks:

- block 1: `A B`
- block 2: `A B`, with C also collected in the balanced A/B/C campaign
- block 3: `B A`, with C also collected

Confirmatory extension:

- block 4: `A B`
- block 5: `B A`

Combined counts:

- A: `n=5`
- B: `n=5`
- C: `n=3`

## Collection validity

- extension runs: `4/4 rc=0`
- combined runs: `13/13 rc=0`
- complete terrain-context coverage
- command validation passed
- state and realized-VX telemetry valid
- work values remain commanded mechanical-work/effort proxies based on
  `tau_cmd * dq`, not battery energy

## Preregistered decision

`DO_NOT_CONFIRM_B_SCHEDULE_SAFETY_OR_LATERAL_PENALTY_V0`

The machine-readable label groups safety and lateral guards together.
In the observed result, every safety and hard-safety rule passed. The
non-confirmation was caused exclusively by two lateral-deviation rules.

## Confirmed performance effects

Five-block paired B-minus-A mean effects:

- mission time: `-6.36405 simulated seconds`
- commanded-work proxy: `-595.02848`
- work-per-meter proxy: `-74.30057`
- realized mean body-frame VX: `+0.00546 m/s`

Repeatability:

- B faster in `4/5` blocks
- B lower commanded-work proxy in `4/5` blocks
- B lower work-per-meter proxy in `4/5` blocks
- B higher realized body-frame VX in `5/5` blocks

All preregistered core and supporting performance criteria passed.

## Failed confirmation criteria

### Maximum lateral deviation

- paired mean B-minus-A penalty: `+0.15465 m`
- preregistered upper limit: `+0.15000 m`
- result: `FAIL`

### Rough-segment lateral displacement

- paired mean B-minus-A penalty: `+0.14781 m`
- preregistered upper limit: `+0.10000 m`
- result: `FAIL`

## Safety result

All safety and hard-safety rules passed:

- paired mean maximum-roll penalty: `+0.52762 deg`
- paired mean maximum-pitch effect: `-0.19060 deg`
- paired mean minimum-base-height effect: `+0.01251 m`
- paired mean VX tracking-error effect: `-0.00047 m/s`
- worst B maximum roll: `7.87294 deg`
- worst B maximum pitch: `8.40859 deg`
- worst B minimum base z: `0.29562 m`

Thus the schedule was not rejected for falling, body-height collapse, or
attitude-guard violation.

## Block-5 diagnostic

Block 5 was the clearest counterexample:

- mission time effect: `-7.0600 s`
- commanded-work effect: `+223.26`
- goal-entry absolute-y penalty: `+0.9304 m`
- maximum absolute-y penalty: `+0.9232 m`
- maximum-roll penalty: `+3.8708 deg`
- minimum-base-height effect: `+0.03775 m`

The schedule remained faster, but lateral behavior deteriorated enough to
prevent promotion.

## Engineering decision

### A

A remains the conservative LL1 default and comparison baseline.

Label:

`valid_authoritative_baseline_world_v5_v0`

### B

B demonstrates repeatable execution, speed, and aggregate commanded-work
benefits, but fails the preregistered lateral confirmation.

It is not promoted to:

- LL1 default,
- final action-bank preferred schedule,
- Objective Selector teacher label,
- learned-policy supervision target.

Label:

`valid_authoritative_performance_gain_but_lateral_confirmation_failed_world_v5_v0`

### C

The earlier fixed-high result remains:

`valid_authoritative_fast_but_stability_variable_not_promoted_world_v5_v0`

## Research interpretation

Because B uses high VX only on flat terrain but shows later rough-terrain
lateral penalties, the evidence motivates testing state/history-sensitive
terrain transitions rather than another identical B campaign.

A plausible next hypothesis is that abrupt context switching at the flat
boundary leaves an unfavorable entry velocity, body state, or gait phase.

The next candidate should therefore be separately preregistered as a new
schedule, for example:

- `vx=0.115` on early flat,
- transition ramp or early reduction to `vx=0.090` before upslope entry,
- `vx=0.090` on difficult terrain.

This is a new hypothesis and must not be treated as a continuation that
overwrites the failed B confirmation.

## Reproducibility artifacts

### Runtime data

- original screening root: `/home/kraken/Tracer/TRACER/logs/revalidation_vx_energy_abc_20260729_131333`
- confirmatory extension root: `/home/kraken/Tracer/TRACER/logs/revalidation_vx_energy_ab_confirm_20260729_150712`
- combined-analysis root: `/home/kraken/Tracer/TRACER/logs/revalidation_vx_energy_ab_combined_20260729_150712`

### Archives

- original raw archive: `/home/kraken/Tracer/TRACER/logs/revalidation_vx_energy_abc_20260729_131333_full.tar.gz`
  - SHA-256: `96225965304d3674f89f3c283336d53715eee1f4467084e5f6d63a65fb2c791a`
- extension raw archive: `/home/kraken/Tracer/TRACER/logs/revalidation_vx_energy_ab_confirm_20260729_150712_full.tar.gz`
  - SHA-256: `eb8c3e56da86e4d55b7a33e5ff14677405489a7a312ae0c46355acf47bcaf2de`
- combined analysis archive: `/home/kraken/Tracer/TRACER/logs/revalidation_vx_energy_ab_combined_20260729_150712_analysis_vx_energy_confirm_v0.tar.gz`
  - SHA-256: `ec28fedeb25dca0db8ee3c5c2816b6f02cd231df0ebe7f6f956b2c8b7ffa736f`

### Frozen scripts and rules

- `/home/kraken/Tracer/TRACER-lowcontroller-audit/scripts/runtime/tracer_run_ll1_vx_energy_ab_confirm_v0.sh`
  - SHA-256: `b443e432983bb76dfdade0ea3b92d421c82ed7a9792ab4ad728a741070a86a26`
- `/home/kraken/Tracer/TRACER-lowcontroller-audit/scripts/analysis/tracer_analyze_ll1_vx_energy_ab_confirm_combined_v0.py`
  - SHA-256: `f11e43b9360ba554f9902648cefda42afeb0f5bb95ee9ebf253bba2a92d193c5`
- `/home/kraken/Tracer/TRACER-lowcontroller-audit/scripts/analysis/tracer_evaluate_ll1_vx_ab_confirm_v0.py`
  - SHA-256: `631a480d0cdd143834f07a014761744f3c4e636341c66328175e905e90da2b2d`
- `/home/kraken/Tracer/TRACER-lowcontroller-audit/reports/phase_ll/ll1_vx_ab_confirm_20260729_150712_preregistered_rules_v0.txt`
  - SHA-256: `c42951d3ecc511fac69b35f98a24e0cc7d04e514bff92e21e1117def858cb5b3`

### Generated evidence

- combined manifest SHA-256: `5ebf3bdb977a16bdaf5e867c3b0044074141a8e20c32864879c35b332788fd7c`
- corrected combined summary SHA-256: `6ee0d6a999574af22568922ee1eb92dabacd381993e0e9f5755bea47104a2c35`
- decision report SHA-256: `051be93f36d629a9357c3b9739114bea6781df50a88e5afd36e1e8abc8c01356`
- rules CSV SHA-256: `747e3df7f2daa4041467aa683166c17cc0b2d940afc0f09bb5616bddacec5d66`
- paired-block CSV SHA-256: `d6453ca6c871e79f7c56e906387523f8d28a4cd619c2242c4059f36538e805e3`

## Next step

Freeze this non-promotion result, then define a separate transition-aware
VX candidate. Do not add post-hoc repeats of the unchanged B schedule to
reverse the preregistered decision.
