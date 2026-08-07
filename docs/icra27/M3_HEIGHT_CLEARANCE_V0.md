# ICRA27 M3 — Body Height × Swing Clearance Characterization v0

## Protocol

Flat MuJoCo / Go2 / PyMPC + WBC.

Fixed:
- vx = 0.20 m/s
- yaw rate = 0
- gait frequency = 1.4 Hz
- duty factor = 0.65

Swept:
- body height = {0.24, 0.28, 0.32} m
- swing clearance = {0.03, 0.06, 0.09} m

Direct-start backend initialization was used so transition robustness was
not mixed into the steady-state authority characterization.

## Viability

All 9 / 9 tested combinations completed the target window without
termination.

Worst observed attitude excursion over the grid:
- max |roll| = 0.67 deg
- max |pitch| = 2.28 deg

## Body-height authority

Linear fits at fixed swing clearance:

- clearance 0.03 m:
  gain = 1.03066, bias = +0.01685 m, R2 = 0.999999
- clearance 0.06 m:
  gain = 1.03154, bias = +0.01529 m, R2 = 0.999997
- clearance 0.09 m:
  gain = 1.02952, bias = +0.01502 m, R2 = 0.999997

Measured body height retains a systematic positive offset, but command
authority is approximately unity and nearly invariant to clearance.

## Swing-clearance authority

Linear fits at fixed body height:

- body height 0.24 m:
  gain = 0.96424, bias = -0.00029 m, R2 = 0.999919
- body height 0.28 m:
  gain = 0.96716, bias = -0.00054 m, R2 = 0.999942
- body height 0.32 m:
  gain = 0.96791, bias = -0.00090 m, R2 = 0.999916

Physical swing clearance therefore tracks the requested clearance with
small attenuation.

## Off-diagonal coupling

Measured body-height variation across the full 0.03--0.09 m clearance
sweep:

- h = 0.24 m: 0.00212 m
- h = 0.28 m: 0.00211 m
- h = 0.32 m: 0.00221 m

Measured-clearance variation across the full 0.24--0.32 m body-height
sweep:

- clearance = 0.03 m: 0.00050 m
- clearance = 0.06 m: 0.00040 m
- clearance = 0.09 m: 0.00028 m

The off-diagonal effects are small relative to the corresponding
diagonal command authority.

## M3 interpretation

Within this tested nominal regime, body height and swing clearance can
be treated as approximately independent continuous high-level action
dimensions.

This differs from frequency and duty factor, whose viability showed
strong joint and transition-history dependence.

This conclusion is local to the tested flat-terrain nominal regime and
must not be interpreted as a global safety guarantee.

## Status

Body height × swing clearance M3 v0 slice: CLOSED.
