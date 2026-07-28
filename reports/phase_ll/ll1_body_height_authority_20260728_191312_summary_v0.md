# LL1 Body-Height Authority Revalidation

## Evidence status

`valid_routing_pass_realized_transient_only_not_repeatable_ll1_v0`

## Purpose

Determine whether body-height commands routed through J7 and the frozen LL1
controller produce a repeatable realized base-height response suitable for the
current high-level action bank.

## Configuration

- World/terrain: flat stationary authority probe.
- Velocity command: 0 m/s.
- Yaw-rate command: 0 rad/s.
- Swing clearance: 0.045 m.
- Controller: frozen LL1.
- Reset policy: one canonical standing reset.
- Runtime policy: one controller, one J7 process, continuous A-B-B-A rollout.
- Preconditioning: 25 seconds at the empirical 0.320 m command.
- A command: body height 0.320 m.
- B command: body height 0.305 m.
- Command difference: -0.015 m.
- Diagnostic J7 envelope for this probe only:
  - maximum absolute body-height delta: 0.020 m;
  - minimum active body height: 0.300 m.
- Repository defaults were not changed.

## J7 routing result

- Projected 0.305 m:
  - accepted;
  - output 0.305 m;
  - 480 decision rows.
- Projected 0.295 m:
  - rejected with `delta_body_h_too_large`;
  - fallback output 0.320 m;
  - 981 decision rows.

J7 command routing and sequence-preserving guarded selection passed.

## Realized response

| Window | Command (m) | Median base z (m) | Drift-corrected effect |
|---|---:|---:|---:|
| A1 | 0.320 | 0.366436940 | — |
| B1 | 0.305 | 0.362663051 | -5.132954 mm |
| B2 | 0.305 | 0.368663277 | -0.219980 mm |
| A2 | 0.320 | 0.370242322 | — |

Additional results:

- A endpoint drift: +3.805382 mm.
- B-half drift under the same 0.305 m command: +6.000226 mm.
- Combined drift-corrected effect: -2.676467 mm.
- Command magnitude: -15.000000 mm.
- Realized/command ratio: 0.178431.
- Minimum settled base z: 0.351570 m.
- Maximum settled absolute roll: 1.069662 degrees.
- Maximum settled absolute pitch: 1.850900 degrees.

## Interpretation

The lower body-height command produced a transient negative base-z response in
B1, but the response was not sustained in B2 despite the same 0.305 m command
remaining active. The B1-to-B2 base-z rise was approximately 6 mm.

Therefore:

- command transport passed;
- J7 guarded routing passed;
- a transient realized response was observed;
- sustained and repeatable realized body-height authority was not established.

The current LL1 body-height dimension must not be promoted into the high-level
action bank.

This result does not prove that body-height control is fundamentally impossible.
The dimension may be revisited after an LL2 redesign with explicit body-height
tracking, support-relative height observation, and synchronized posture/gait
control.

## Current action-bank decision

- Preserve clearance 0.045 m as the main/default candidate.
- Retain clearance 0.055 m only as an authoritative but performance-rejected
  diagnostic candidate.
- Exclude body height from the current LL1 action bank.
- Move the next authority study to the forward-velocity dimension.
- Defer body-height reintroduction to LL2.

## Reproducibility

- Executed large-signal harness:
  `scripts/analysis/tracer_run_j7_body_height_large_signal_abba_authority_v0.sh`
- Offline analyzer:
  `scripts/analysis/tracer_analyze_j7_body_height_large_signal_abba_v0.py`
- Small-signal result:
  `/tmp/tracer_j7_body_height_interleaved_abba_authority_20260728_180351`
- Large-signal result:
  `/tmp/tracer_j7_body_height_large_signal_abba_authority_20260728_191312`
- Preserved result archive:
  `/home/kraken/Tracer/TRACER/logs/tracer_ll1_body_height_authority_20260728_191312_full.tar.gz`

SHA-256:

- archive: `5a0a8ce183341d8850bd9953e131223aef71b36e303da16ad462b760b2e21904`
- harness: `56dd26ef4cfb413a52672a1d5a717181ec8ed448709f3617b46f5737cd95b187`
- analyzer: `30646ba7bbcd1ebe6e44658cb0a1b3119e4da380b1d5292ad89f661b01fdc24e`
- corrected summary: `343bb0c0bf1f0c7efd83f95f962134e92a35b613d7f4050492c2d94ba91cb327`
- window summary: `7e102a5412b39af9259fc1191a6f938209b700d0305caef7d2264a77f9cf41ca`
- realized CSV: `24a15978991eed41406038543e2d33fb2ce10dd452723f2f7d59a314cb9425d5`
- J7 decision CSV: `34d6727da2477cf1c7eea530e73cb71babe11a83a282659301b898bec2467acb`

## Scope guard

This evidence does not establish terrain benefit, stability benefit, energy
benefit, generalization, Objective Selector readiness, or learned-policy
performance.
