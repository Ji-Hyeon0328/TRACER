# TRACER Phase-C Mixed Terrain A/B/C Summary v0

## Setup

- World: `tracer_mixed_solid_course_v0`
- Low-level controller: frozen A1-QP-MPC Gazebo controller
- Runtime stack:
  - qwer standing reset
  - ROS2 `/tracer/mpc_reference`
  - ROS1/ROS2 UDP bridge
  - model-state odom bridge
  - oracle terrain context provider
- Mixed course order:
  - flat
  - upslope
  - rough
  - downslope
  - goal_flat

## Result

| Policy | Success | Goal-zone time [s] | Max lateral drift \|y\| [m] | Final x [m] | Final y [m] |
|---|---:|---:|---:|---:|---:|
| Fixed single command | True | 240.00 | 0.711 | 8.410 | -0.690 |
| Oracle terrain schedule | True | 294.00 | 0.321 | 8.021 | 0.310 |
| Beta-aware schedule | True | 348.00 | 0.190 | 8.010 | 0.096 |

## Interpretation

The fixed single-command baseline completes the course fastest, but it accumulates the largest lateral drift.

The oracle terrain schedule slows down on rough and downhill regions, reducing lateral drift substantially.

The beta-aware schedule explicitly increases the stability weight on difficult terrain and produces the smallest lateral drift, at the cost of longer traversal time.

This validates the Phase-C high-level interface:

```text
terrain/context
→ objective weights β
→ meta-level command schedule
→ low-level MPC reference
→ frozen A1-QP-MPC controller
## Current takeaway

The Phase-C scaffold supports a clear speed–stability tradeoff:
fixed: fastest, least stable
oracle: medium speed, more stable
beta-aware: slowest, most stable
