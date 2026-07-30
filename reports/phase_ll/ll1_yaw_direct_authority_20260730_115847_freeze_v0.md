# LL1 Direct Signed Yaw Authority Freeze

## Evidence identity

- Runtime root: `/home/kraken/Tracer/TRACER/logs/smoke_direct_signed_yaw_authority_20260730_115847`
- Full archive: `/home/kraken/Tracer/TRACER/logs/smoke_direct_signed_yaw_authority_20260730_115847.tar.gz`
- Archive SHA-256: `dc3d83844c6299b9db26a1027186bfefb77eddddb2d28c3a156f1a0d9acdc6e7`
- Source branch before freeze: `TRACER-lowcontroller`
- Source head before freeze: `7e3ca7b5eff229aeac4b81d9c2178701bc259997`

## Experimental configuration

- World: `tracer_mixed_solid_course_v0`
- Fixed forward velocity: `0.090 m/s`
- Fixed body-height command: `0.320 m`
- Fixed swing-clearance command: `0.045 m`
- Signed yaw commands: `+0.050 rad/s`, `-0.050 rad/s`
- Route: ROS2 publisher → UDP → ROS1 `/tracer/mpc_reference` → active LL1
- J4/J7: bypassed
- Sequence: Z0 → P → Z1 → N → Z2
- Selector timing: odom-fresh active wall time
- Duration validation: realized ROS1/Gazebo simulation time

## Result

- Classification: `valid_direct_signed_yaw_authority_smoke_v0`
- Transport pass: true
- Duration pass: true
- Authority pass: true
- Safety pass: true

### Drift-corrected realized effects

- Positive angular-velocity effect: `+0.044815043 rad/s`
- Negative angular-velocity effect: `-0.054206991 rad/s`
- Positive unwrapped-yaw slope effect: `+0.043902864 rad/s`
- Negative unwrapped-yaw slope effect: `-0.055482892 rad/s`

### Signed heading response

- Positive segment heading delta: `+0.217512952 rad`
- Negative segment heading delta: `-0.247006565 rad`

### Safety observations

- Worst absolute roll: `2.531765364 deg`
- Worst absolute pitch: `4.793632927 deg`
- Minimum base z: `0.310533141 m`

## Valid claim

The frozen LL1 controller has direct, signed, realized yaw-rate authority for
`±0.050 rad/s` commands under the tested world-v0 runtime.

## Claims not established

- No lateral-path performance claim
- No terrain-generalization claim
- No energy-performance claim
- No J4/J7 guarded-route claim
- No Objective Selector or learned-planner claim
- No world-v5 performance claim

## Next gate

Audit and test the guarded J7 route at the allowed yaw increment boundary,
starting with `0 → +0.020 → 0 → -0.020 → 0 rad/s`.
