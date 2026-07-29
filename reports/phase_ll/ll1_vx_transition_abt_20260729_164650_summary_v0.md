# LL1 VX Transition-Aware Hypothesis Non-Promotion Freeze

## Scope

This record freezes the balanced A/B/T screening of a transition-aware
VX schedule in `tracer_mixed_stress_course_v5_lowfric_from_solid`.

Base audit commit:

`3afa58711106e943fd7c450ed8335ba74b302801`

This is descriptive world-v5 engineering evidence. It is not evidence of
cross-world generalization, battery-energy optimality, or a final
Objective Selector teacher label.

## Conditions

### A — fixed low

- `vx=0.090 m/s` throughout the active course

### B — abrupt terrain schedule

- flat/start-flat: `vx=0.115 m/s`
- difficult terrain: `vx=0.090 m/s`
- abrupt reduction at the terrain-label boundary

### T — transition-aware schedule

- `x <= 1.40 m`: `vx=0.115 m/s`
- `1.40 < x < 1.90 m`: linear reduction to `0.090 m/s`
- `x >= 1.90 m`: `vx=0.090 m/s`
- difficult terrain: `vx=0.090 m/s`

Fixed parameters:

- body height: `0.320 m`
- swing clearance: `0.045 m`
- yaw command: `0`
- frozen LL1 A1-QP-MPC

## Collection validity

- balanced order: `ABT / TAB / BTA`
- A/B/T counts: `3/3/3`
- rollout return codes: `9/9 rc=0`
- full-course command and telemetry validation: `9/9 PASS`
- transition-ramp implementation validation: `3/3 PASS`

## Preregistered decision

`DO_NOT_PROMOTE_TRANSITION_AWARE_INSUFFICIENT_BASELINE_TRADEOFF_WORLD_V5_V0`

The T schedule passed mission-time, commanded-work, transition-cost, and
hard-safety requirements. It failed all four preregistered lateral
requirements.

## T relative to A

Paired mean effects:

- mission time: `-6.44000 simulated seconds` — PASS
- commanded-work proxy: `-689.39880` — PASS
- realized body-frame VX: `+0.00457 m/s`
- maximum lateral-deviation penalty: `+0.23738 m` — FAIL
- rough lateral-displacement penalty: `+0.15692 m` — FAIL

T retained the useful time/work benefit associated with using higher VX
on early flat terrain, but did not retain the required lateral behavior
relative to conservative A.

## T relative to abrupt B

Paired mean effects:

- mission-time penalty: `+1.06667 simulated seconds` — PASS
- commanded-work penalty: `+65.84662` — PASS
- maximum lateral-deviation effect: `+0.20419 m` — FAIL
- rough lateral-displacement effect: `+0.03283 m` — FAIL

The preregistered hypothesis required improvements of at least `0.05 m`
in both lateral metrics. Neither improvement occurred.

## Terrain-entry-state result

T did modify terrain-entry state relative to abrupt B:

- upslope-entry body VX: `-0.01472 m/s`
- upslope-entry absolute lateral position: `-0.11565 m`
- rough-entry absolute lateral position: `-0.11340 m`

Therefore, the pre-boundary deceleration intervention was executed and
did affect entry state. However, that state change did not produce a
repeatable downstream lateral benefit.

## Variability diagnostic

The second T rollout was the principal counterexample:

- goal-entry absolute y: `1.8885 m`
- maximum absolute y: `1.8885 m`
- rough absolute delta-y: `0.6162 m`

This prevents promotion even though the other T rollouts were less
laterally extreme.

## Safety

T hard-safety results passed:

- worst T maximum roll: `6.63691 deg`
- worst T maximum pitch: `8.30129 deg`
- worst T minimum base height: `0.29741 m`

Thus the T schedule is not rejected for falling, attitude-guard
violation, or base-height collapse.

## Engineering decision

The hand-designed VX-transition hypothesis is closed for LL1.

Do not continue with:

- additional ramp-start or ramp-end tuning,
- alternative interpolation functions,
- post-hoc repeats of unchanged T,
- promotion of T into the final action bank,
- use of T as an Objective Selector teacher label.

Labels:

- A:
  `valid_authoritative_baseline_world_v5_v0`
- B:
  `valid_authoritative_performance_gain_but_lateral_confirmation_failed_world_v5_v0`
- T:
  `valid_authoritative_transition_executed_but_lateral_hypothesis_not_supported_world_v5_v0`

## Research implication

VX scheduling alone does not provide sufficient lateral authority in the
present frozen LL1 configuration.

The next high-level action-axis validation should examine yaw-rate or
heading correction. Its first requirement is physical and repeatable
yaw authority, followed by a lateral-correction performance test.

## Reproducibility artifacts

### Runtime

- run root: `/home/kraken/Tracer/TRACER/logs/revalidation_vx_transition_abt_20260729_164650`
- raw archive: `/home/kraken/Tracer/TRACER/logs/revalidation_vx_transition_abt_20260729_164650_full.tar.gz`
  - SHA-256: `ab556c769e1d24ba414961582eb2841bc7223ae396a2fcbb80dcca548639b0bf`
- analysis archive: `/home/kraken/Tracer/TRACER/logs/revalidation_vx_transition_abt_20260729_164650_analysis_vx_transition_abt_v0.tar.gz`
  - SHA-256: `8f67846b579d3ada12d6c0ddadceb17bae336016e0ae9079f9510ae6006f171e`

### Frozen scripts

- selector: `/home/kraken/Tracer/TRACER-lowcontroller-audit/scripts/runtime/tracer_phase_d4_transition_aware_selector_node_v0.py`
  - SHA-256: `1ef2a6e355943accad6eb355997a0a961533bff4ac94c0fbe67dd049a43677a0`
- start script: `/home/kraken/Tracer/TRACER-lowcontroller-audit/scripts/runtime/tracer_start_phase_d4_transition_ll1_energy_v0.sh`
  - SHA-256: `84015cf5e07c3cadbca30e47df916863242d90845176dc99455ce72cbb4bf919`
- campaign: `/home/kraken/Tracer/TRACER-lowcontroller-audit/scripts/runtime/tracer_run_ll1_vx_transition_abt_balanced_v0.sh`
  - SHA-256: `7fe3b32a59ec99164fa7c233053a43757491cffbfd505bef241b51c4573f02a6`
- analyzer: `/home/kraken/Tracer/TRACER-lowcontroller-audit/scripts/analysis/tracer_analyze_ll1_vx_transition_abt_v0.py`
  - SHA-256: `8746015c6e6a381f6c67d9408d4a9c3f9cbd6c4d5777d3529d0fa895b28483d7`
- preregistration: `/home/kraken/Tracer/TRACER-lowcontroller-audit/reports/phase_ll/ll1_vx_transition_abt_20260729_164650_preregistered_v0.txt`
  - SHA-256: `e605d973ec04cf9b277483a4bf6a0e4532bac3096781e652b3bead033d14ae2a`

### Generated evidence

- generated summary SHA-256: `92807c5c756fd3f90964f08f96c2d21bfbfe7cb07b34c6d6027b71e71ca8e685`
- decision JSON SHA-256: `b2bcb843517c50fa8f6690a2040f25cec6ed1eef842236c87e4d417b479e8621`
- rules CSV SHA-256: `5b39e9da45a0fbf8bdf28dc74e77f17c17ccfa30c168570bbffa8ed4e3f0fa05`
- paired-effects CSV SHA-256: `dcf48df5e3edaa8b58f41409840f5d3e0d1e49462360858ac51fec1380b64feb`
- run metrics SHA-256: `70e71c71f2fdbbc6192fa02116d29086f195572b2e4b69dac3afc5abf4d35def`
- manifest SHA-256: `4a93781bf8f12a2e8df8adbf64c83029948abe6ec307e603cb569908d4657e01`
