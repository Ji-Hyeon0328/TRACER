# LL1 Realized Swing-Apex Authority A/B

## Purpose

Verify that the feature-gated TRACER swing-apex residual not only changes
the commanded swing target, but also produces a corresponding change in
the realized foot position.

## Controller checkpoints

- `b9cab78`: feature-gated TRACER swing-apex residual
- `85adb97`: fail closed on invalid residual inputs and configuration

## Conditions

Two independent trials were initialized using the same canonical standing
reset procedure.

| Trial | Clearance command | Expected apex residual |
|---|---:|---:|
| Neutral | 0.045 m | 0 mm |
| High | 0.060 m | +15 mm |

The command was preloaded while Gazebo physics was paused. Each trial used
a 3-second warmup followed by an 8-second collection interval.

The matched-phase analysis compared the same leg and swing-phase bins in
the approximate apex region.

## Target-level validation

| Trial | Maximum shape | Maximum applied bump |
|---|---:|---:|
| Neutral 0.045 | 1.0 | 0 mm |
| High 0.060 | 1.0 | +15 mm |

For both trials:

- residual active flag was correct;
- bounded delta matched the configured command;
- applied bump equaled `delta * shape`;
- stance bump remained zero.

## Matched-phase realized result

| Leg | Target shift | Actual foot-position shift |
|---|---:|---:|
| FL | +11.12 mm | +11.45 mm |
| FR | +12.16 mm | +11.43 mm |
| RL | +12.08 mm | +11.23 mm |
| RR | +11.12 mm | +12.01 mm |

Combined results:

- median target shift: `+11.360 mm`
- median actual shift: `+11.439 mm`
- positive actual-shift cells: `100%`
- median realized/target ratio: `1.007`
- median tracking-error change: `-0.199 mm`

The matched target shift is smaller than the full +15 mm apex residual
because the analysis aggregates phase bins around the apex rather than
only the exact `phase=0.5` sample.

## Conclusion

The high-clearance command produced a consistent positive realized
foot-height shift across all four legs. The realized shift closely
matched the corresponding target shift.

This establishes low-level physical authority of the TRACER clearance
command. It does not yet establish improved terrain traversal,
stability, or task performance.

## Files

- `neutral_045.csv`
- `high_060.csv`
- `matched_phase_summary.txt`
- `../../../scripts/analysis/tracer_ll1_realized_authority_ab_v0.sh`
