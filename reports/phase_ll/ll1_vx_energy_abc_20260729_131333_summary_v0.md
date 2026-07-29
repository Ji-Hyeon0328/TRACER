# LL1 VX Energy A/B/C Balanced Screening Freeze

## Scope

This record freezes the balanced LL1 VX performance-screening campaign
conducted in `tracer_mixed_stress_course_v5_lowfric_from_solid`.

This is performance-screening evidence for executable VX commands whose
low-level authority had already been established independently.

It is not evidence of:

- battery electrical-energy optimality,
- cross-world generalization,
- a final Objective Selector teacher label,
- statistical superiority beyond the present balanced `n=3` screening,
- suitability of fixed-high VX on every terrain.

## Base state

- audit branch: `TRACER-lowcontroller`
- base commit: `145cbe8b6526184ba198050ccebadc7a351385c1`
- runtime routing: direct Phase-D4 context selector to `/tracer/mpc_reference`
- fixed body height: `0.320 m`
- fixed swing clearance: `0.045 m`
- yaw command: `0`
- world: `tracer_mixed_stress_course_v5_lowfric_from_solid`

## Conditions

### A — fixed low

- all active terrain contexts: `vx=0.090 m/s`

### B — terrain scheduled

- flat/start-flat: `vx=0.115 m/s`
- upslope/rough/downslope: `vx=0.090 m/s`
- unknown: fail closed to `vx=0.090 m/s`

### C — fixed high

- all active terrain contexts: `vx=0.115 m/s`

Balanced order:

- block 1: `A B C`
- block 2: `C A B`
- block 3: `B C A`

## Collection validity

- total runs: `9`
- A/B/C: `3/3/3`
- rollout return codes: `9/9 rc=0`
- full-course command/context validation: `9/9 PASS`
- logger validity fraction: `1.0` in every run
- post-goal hold excluded from mission analysis
- work values are commanded mechanical-work/effort proxies based on
  `tau_cmd * dq`, not battery electrical energy.

## Condition-level results

| Condition | Goal time [sim s] | Goal abs y [m] | Max abs y [m] | Work proxy | Work/m proxy | Mean body VX [m/s] | Max roll [deg] | Max pitch [deg] | Min z [m] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A fixed-low | 113.220 ± 1.800 | 0.619 ± 0.120 | 0.631 ± 0.118 | 12091.3 ± 274.2 | 1511.5 ± 34.4 | 0.0668 ± 0.0004 | 3.638 ± 0.998 | 8.216 ± 0.345 | 0.314 ± 0.011 |
| B scheduled | 107.247 ± 4.279 | 0.538 ± 0.399 | 0.684 ± 0.263 | 11324.0 ± 283.5 | 1415.6 ± 35.2 | 0.0726 ± 0.0010 | 2.852 ± 0.382 | 8.157 ± 0.312 | 0.318 ± 0.010 |
| C fixed-high | 88.100 ± 2.575 | 0.656 ± 0.820 | 0.733 ± 0.773 | 9577.2 ± 125.8 | 1197.2 ± 15.7 | 0.0883 ± 0.0015 | 6.129 ± 5.334 | 8.764 ± 1.082 | 0.302 ± 0.003 |

## Paired interpretation

### B minus A

- goal time: `-5.973 s` mean, approximately `-5.23%`
- commanded-work proxy: `-767.24`, lower in all three blocks
- work-per-meter proxy: `-95.90`, lower in all three blocks
- mean realized body VX: `+0.00574 m/s`, higher in all three blocks
- mean absolute tracking error: `-0.00107 m/s`, lower in all three blocks
- minimum base height: `+0.00389 m`
- lateral metrics were mixed and did not establish consistent improvement
- goal-time improvement occurred in two blocks, with a small reversal in one

Decision:

`B` is the screening-preferred terrain-scheduled VX candidate, but is not
yet a final promoted default or a final learned-policy supervision target.

Label:

`valid_authoritative_screening_preferred_vx_schedule_world_v5_v0`

### C minus A

- goal time: `-25.120 s`, faster in all three blocks
- commanded-work proxy: `-2514.10`, lower in all three blocks
- mean realized body VX: `+0.02152 m/s`, higher in all three blocks
- minimum base height: `-0.01225 m`, lower in all three blocks
- roll and lateral outcomes showed large variability
- one run reached approximately `12.28 deg` maximum roll and
  `1.595 m` goal-entry absolute lateral displacement

Decision:

Fixed-high `C` is not promoted as an all-terrain default. The
`0.115 m/s` command remains a valid executable flat-speed candidate and
is retained through condition `B`.

Label:

`valid_authoritative_fast_but_stability_variable_not_promoted_world_v5_v0`

### A

`A` remains the conservative LL1 comparison baseline.

Label:

`valid_authoritative_baseline_world_v5_v0`

## Main research interpretation

The campaign supports the engineering value of terrain-conditioned VX
scheduling in this world:

- high VX on flat terrain captures useful traversal-time benefit,
- low VX on difficult terrain avoids part of the stability variability
  observed under fixed-high traversal,
- the scheduled policy presents a compromise between conservative
  fixed-low and aggressive fixed-high operation.

This is balanced descriptive screening evidence only. Confirmatory paired
repeats and cross-world validation are required before final action-bank,
Objective Selector, or policy-label promotion.

## Reproducibility artifacts

- runtime run root: `/home/kraken/Tracer/TRACER/logs/revalidation_vx_energy_abc_20260729_131333`
- runtime run root resolved: `/mnt/share/nas/Yoo/Tracer/TRACER/logs/revalidation_vx_energy_abc_20260729_131333`
- full archive: `/home/kraken/Tracer/TRACER/logs/revalidation_vx_energy_abc_20260729_131333_full.tar.gz`
- full archive SHA-256: `96225965304d3674f89f3c283336d53715eee1f4467084e5f6d63a65fb2c791a`
- analysis archive: `/home/kraken/Tracer/TRACER/logs/revalidation_vx_energy_abc_20260729_131333_analysis_vx_energy_v0.tar.gz`
- analysis archive SHA-256: `bf3256940342874f2fcfe0e2d3d941b3bf47991d7f43a8ce7532b004d3b1e6a6`

### Frozen scripts

- `/home/kraken/Tracer/TRACER-lowcontroller-audit/scripts/runtime/tracer_run_ll1_vx_energy_abc_balanced_v0.sh`
  - SHA-256: `4f597c0ad99d627c3d0add4d4722959646b18173f22e78216e2178ad840fdc44`
- `/home/kraken/Tracer/TRACER-lowcontroller-audit/scripts/analysis/tracer_analyze_ll1_vx_energy_abc_balanced_v0.py`
  - SHA-256: `6f56849e8ead9a23b199c655e545e9b7610d70d1820d0d59c06f2d5d9dc271e5`
- `/home/kraken/Tracer/TRACER-lowcontroller-audit/scripts/runtime/tracer_ros1_joint_energy_vx_logger_v0.py`
  - SHA-256: `999b91b06629a43770cd4c1231b8da969869145484e2e20f12985c5dfbeae1b3`

### Source evidence

- manifest SHA-256: `411e34a00e4ec71165a42e71f57d013fafc0be3e0c457a541179358a3ea37ce6`
- generated summary SHA-256: `cc3b2c7187caa9bfaf0cb721398b3eeeecbd3c502ad74246f457181a72675af8`
- condition summary SHA-256: `2217421a8c60095eab96f07c79d83c8a9429bf11f1959fd3f78c627f23d7d98a`
- paired effects SHA-256: `4e7a67c9dbd51d23b647097ca4b416086aee5d4eb9b2e7fadb7bd796fdf4eaa6`

## Next required validation

Extend A/B comparison by two additional balanced blocks:

- block 4: `A B`
- block 5: `B A`

This raises A and B to `n=5` while balancing execution order. The next
decision should require repeatable mission-time/work improvement without
a meaningful safety or lateral-deviation penalty.
