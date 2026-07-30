# Phase-B LL1 Bank–J7 Numerical Contract Correction v2

## Finding

The frozen J7 source uses a default maximum swing-clearance deviation of
`0.006 m` around the empirical reference. The historical candidate bank v0
used clearances `0.035, 0.045, 0.055 m` around a `0.045 m` empirical anchor.
The two extreme actions therefore requested `±0.010 m` and would be rejected
with `delta_clearance_too_large`.

The v0/v1 static contract files incorrectly modeled the clearance guard as
`0.010 m`. They remain preserved as historical audit artifacts but must not
be used for active runtime.

## Correction

The runtime bank v1 uses:

- forward speed: `0.040, 0.065, 0.090 m/s`;
- swing clearance: `0.040, 0.045, 0.050 m`;
- fixed body height: `0.320 m`;
- deterministic guarded yaw: `|yaw_rate| <= 0.020 rad/s`.

Around the empirical anchor `vx=0.065 m/s`, `clearance=0.045 m`, the largest
deviations are:

- `|delta vx| = 0.025 < 0.030 m/s`;
- `|delta clearance| = 0.005 < 0.006 m`;
- `|delta body height| = 0.000 < 0.006 m`;
- `|delta yaw| = 0.020 <= 0.020 rad/s`.

The v1 semantic split remains:

- `theta[0] = 1`: active J7 gate token;
- `theta[1] = 0..8`: true bank action ID;
- `theta[8]`: hold override.

## Runtime status

This correction proves static compatibility only. The nine actions remain
candidates until the flat Gazebo feasibility sweep confirms active J7
acceptance, ROS1 command transport, and realized locomotion safety.
