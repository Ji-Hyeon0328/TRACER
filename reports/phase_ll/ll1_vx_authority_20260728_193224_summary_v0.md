# LL1 Forward-Velocity Authority Revalidation

## Evidence status

`valid_authoritative_repeatable_realized_vx_response_ll1_v0`

## Purpose

Determine whether forward-velocity references routed through J7 and the frozen
LL1 controller produce distinguishable, repeatable realized body-frame
velocities suitable for use as high-level action-bank candidates.

## Configuration

- World: `tracer_mixed_stress_course_v5_lowfric_from_solid.world`.
- Authority region: start-flat segment only.
- Flat-region runtime guard: base x below 1.8 m.
- Empirical A command: 0.090 m/s.
- Accepted B command: 0.115 m/s.
- Command difference: +0.025 m/s.
- Rejected sentinel: 0.125 m/s, difference +0.035 m/s.
- J7 default maximum absolute velocity delta: 0.030 m/s.
- Body-height command: 0.320 m.
- Swing-clearance command: 0.045 m.
- Yaw-rate command: 0 rad/s.
- One canonical reset, one controller process, and one continuous J7 process.
- A1-B1-B2-A2 interleaved schedule.

Both A and B commands remain above the LL1 0.05 m/s position-lock transition.

## Collection note

The initial harness returned nonzero after collection because its validation
expected at least 40 samples per 2.5-second window. Gazebo model states were
observed at approximately 11 to 13 Hz, producing 31 or 32 samples per window.

The observer completed all four windows and wrote 125 valid rows. The raw CSV
was recovered from the LL1 container. Offline analysis used the same settled
time intervals and authority thresholds, with the minimum settled-row
requirement corrected from 30 to 10.

This was a post-collection validation mismatch, not an interrupted rollout.

## J7 routing

- Projected 0.115 m/s:
  - output 0.115 m/s;
  - accepted with `accepted_guarded`;
  - projected source;
  - 110 rows.
- Projected 0.125 m/s:
  - output fallback 0.090 m/s;
  - rejected with `delta_vx_too_large`;
  - empirical source;
  - 206 rows.

Sequence-preserving J7 command routing passed.

## Realized response

| Window | Command | Samples | Median body vx | Drift-corrected effect |
|---|---:|---:|---:|---:|
| A1 | 0.090 m/s | 17 | 0.080018882 m/s | — |
| B1 | 0.115 m/s | 18 | 0.080185441 m/s | +0.006224403 m/s |
| B2 | 0.115 m/s | 17 | 0.087952007 m/s | +0.019039174 m/s |
| A2 | 0.090 m/s | 18 | 0.062854989 m/s | — |

Additional results:

- A endpoint velocity drift: -0.017163892 m/s.
- B-half velocity drift: +0.007766566 m/s.
- Simple ABBA effect: +0.012631788 m/s.
- Combined drift-corrected effect: +0.012631788 m/s.
- Command difference: +0.025000000 m/s.
- Realized-to-command ratio: 0.505272.
- Maximum settled x: 0.280476 m.
- Minimum settled base z: 0.328246 m.
- Maximum settled absolute roll: 1.116282 degrees.
- Maximum settled absolute pitch: 2.684259 degrees.

## Interpretation

Both B windows produced positive drift-corrected body-frame velocity responses:

- B1 exceeded the predeclared +0.005 m/s per-window threshold.
- B2 exceeded the predeclared +0.005 m/s per-window threshold.
- Their combined response exceeded the +0.008 m/s combined threshold.

Therefore, 0.090 m/s and 0.115 m/s are distinguishable executable velocity
actions in the current frozen LL1 stack.

The response is modest and baseline drift is non-negligible. This evidence
establishes command authority, not accurate velocity tracking or superiority of
the higher command.

## Current action-bank decision

- Preserve swing clearance 0.045 m as the default clearance.
- Do not promote swing clearance 0.055 m on world-v5 performance evidence.
- Exclude body height from the current LL1 action bank.
- Retain 0.090 m/s and 0.115 m/s as authority-certified velocity candidates.
- Treat approximately 0.040 m/s as a separate near-goal/position-lock regime.
- Evaluate velocity candidate performance with balanced mission rollouts before
  using either candidate as terrain-optimal supervision.

## Reproducibility

Repository files:

- corrected reproducibility harness:
  `scripts/analysis/tracer_run_j7_vx_interleaved_abba_authority_v0.sh`
- corrected offline analyzer:
  `scripts/analysis/tracer_analyze_j7_vx_interleaved_abba_authority_v0.py`

Exact executed files and raw evidence are preserved in:

- `/home/kraken/Tracer/TRACER/logs/tracer_ll1_vx_authority_20260728_193224_full.tar.gz`

Original result directory:

- `/tmp/tracer_j7_vx_interleaved_abba_authority_20260728_193224`

SHA-256:

- archive: `1e5bdcdd45a267e88d87ba00e297d6d30fb3000fc3c6ae3ccd83ee821f6d1cb7`
- exactly executed harness: `b13f4e110efcc7a2a3c9440dd46e50b1711d4535475e806546113b75ab664345`
- exactly embedded executed analyzer: `ed99ded1cf4d2b58226dce75c6cffbf0040b44a2841053b188646c66441de0f5`
- offline threshold-corrected analyzer: `70abec7f55f6c44f6570fab1393563ac3281a4ba5b6581cb3f67a1ade834d796`
- repository corrected harness: `0bf984da957176c1e73ea97a22f4164ee5735103caad8f31d5727c9a22992db1`
- repository corrected analyzer: `70abec7f55f6c44f6570fab1393563ac3281a4ba5b6581cb3f67a1ade834d796`
- summary: `d54f79af38deee8717d99e71ab7c8d557b906766003e476fb4ad989dd642c3c6`
- window table: `0dd10559cd24b10b34eb55debb5524e95416b0580d19e197ee08e412cc25ba8e`
- realized CSV: `63d977de2bf0fb24d9cdeebaa8aecfd133541608b148e3c4ccc611e75df9e499`
- J7 decision CSV: `9c2b37195c402538b3e80492028938d3e2b66cd3068dab3869378bbc5476aabd`

## Scope guard

This result establishes forward-velocity command authority only. It does not
establish speed optimality, time improvement, terrain benefit, stability
benefit, energy benefit, generalization, Objective Selector readiness, or
learned-policy performance.
