# J7 Native-Relative Realized Swing-Foot Response

## Purpose

This result evaluates whether a clearance residual selected by the J7
guard reaches the low-level controller and produces an observable
swing-foot response.

The experiment was executed as two reset-separated trials within one
harness invocation.

## Trial definitions

### Accepted trial

```text
trial: accepted_050
empirical clearance: 0.045 m
projected clearance: 0.050 m
J7 decision: accepted_guarded
J7 source: projected
J7 output: 0.050 m
LL1 residual: +0.005 m
'''
## Rejected trial
'''
trial: rejected_060
empirical clearance: 0.045 m
projected clearance: 0.060 m
J7 decision: delta_clearance_too_large
J7 source: empirical
J7 output: 0.045 m
LL1 residual: 0 m
'''
Each trial produced:

161 ROS1 reference packets;
98 unique LL1 debug packets;
392 realized-foot rows;
both swing and stance observations;
a normalized swing shape reaching 1.0;
zero command, residual, and bump-relation errors.
Initial absolute comparison

The initial matched-phase comparison used absolute target_z and
actual_z.

Across 12 matched apex cells:

median native-target difference:  -6.654658 mm
median residual-bump difference:  +4.608000 mm
median final-target difference:   -1.654658 mm
median absolute actual difference: -3.188466 mm

Only 16.7% of absolute actual-height cells were positive.

This does not indicate failure of the J7 or LL1 residual path. The
reset-separated trials had an opposing native swing-trajectory
difference larger than the selected residual.

Native-relative decomposition

The native target was reconstructed exactly as:

native_target_z = target_z - applied_bump

The actual response relative to that native target was then computed
as:

realized_relative_z = actual_z - native_target_z

The cell-wise identities were:

target difference
= native-target difference + bump difference

actual difference
= native-target difference + native-relative realized difference

Maximum decomposition and debug-relation errors were on the order of
1e-17.

Native-relative result

Across the same 12 matched apex cells and all four legs:

median residual-bump difference:
+4.608000 mm

median native-relative actual response:
+3.678587 mm

positive residual-bump cells:
100%

positive native-relative realized cells:
100%

Per-leg median native-relative responses were:

FL: +4.306080 mm
FR: +2.694253 mm
RL: +2.735370 mm
RR: +4.236205 mm
Supported conclusion

The J7-selected residual reached LL1 and produced a consistently
positive actual swing-foot response relative to each trial's
reconstructed native swing trajectory.

Important qualification

The native-relative analysis was introduced after the original
absolute comparison failed because of the discovered native-trajectory
mismatch. It is therefore post-hoc diagnostic evidence rather than a
pre-registered confirmatory comparison.

This result does not establish that the accepted trial had a higher
absolute world-frame or body-frame foot position.

Not established

This experiment does not establish:

absolute foot-height increase across reset-separated trials;
terrain-performance improvement;
stability improvement;
energy improvement;
learned Objective Selector readiness;
learned RL/meta-planner readiness.

A stronger absolute comparison would require a synchronized or
interleaved design that holds the native swing trajectory sufficiently
constant between accepted and empirical conditions.
