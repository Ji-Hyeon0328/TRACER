# J7-to-LL1 Guarded Clearance Authority

This directory freezes the end-to-end validation of the active J7
guarded-reference path through the frozen A1-QP-MPC low-level controller
interface.

## Validated path

```text
empirical/projected/theta/context inputs
→ J7 guarded decision
→ ROS2 /tracer/mpc_reference
→ UDP bridge
→ ROS1 /tracer/mpc_reference
→ LL1 swing-apex residual
Accepted case

The empirical clearance was 0.045 m and the projected clearance was
0.050 m.

J7 produced:

j7_accept = 1
j7_reason = accepted_guarded
j7_source = projected

The selected 0.050 m command reached ROS1 unchanged and produced an
LL1 residual of +0.005 m. The maximum observed applied bump was
0.005000000 m.

Rejected case

The empirical clearance was 0.045 m and the projected clearance was
0.060 m.

J7 produced:

j7_accept = 0
j7_reason = delta_clearance_too_large
j7_source = empirical

The empirical fallback 0.045 m reached ROS1 and produced an LL1
residual of 0 m.

Interpretation

This experiment directly proves same-run guarded target authority from
J7 through the ROS2/UDP/ROS1 transport and into the LL1 swing target.

It does not independently measure same-run physical foot-height
displacement. Realized low-level authority was previously established
by the matched-phase LL1 experiment frozen in commit f6fb596.

This result also does not establish terrain benefit, stability benefit,
energy benefit, or learned selector/policy readiness.

Runtime safety

After completion:

J7 active source was stopped.
The LL1 controller was stopped.
The ROS2 sender and ROS1 receiver were stopped.
The swing-apex residual feature was restored to false.
Gazebo physics was paused.
No joint command publisher remained.
