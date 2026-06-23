# TRACER Fusion Policy V1 Decisions - 2026-06-23

## Confirmed integration
- Fusion policy publishes `/tracer/mpc_reference`.
- Fusion policy publishes `/tracer/objective_weights`.
- ROS2 -> UDP -> ROS1 bridge delivers command to A1-QP-MPC.
- Gazebo settle wait is required after world relaunch before reset.

## Positive cases
### flat_normal
- Policy: fast
- Command: vx=0.28, body_height=0.295, clearance=0.03, enable=1
- Result: valid forward motion.

### sponge_firm_downslope_5deg_forward
- Policy: sponge_tall_10_c080
- Command: vx=0.10, body_height=0.335, clearance=0.08, enable=1
- Absolute min z looked low, but terrain-relative min z was about 0.19.
- Relative fall heuristic: false.
- Decision: keep sponge_tall_10_c080 as positive intermediate locomotion case.

## Negative / no-simple-primitive cases
### slippery_mid_downslope_5deg_forward_postfix
Tested:
- cautious
- slip_crawl_06_lowclear
- safe_stop
- active_hold

Results:
- cautious/slip_crawl failed.
- safe_stop still slid/fell.
- active_hold was better than passive stop but still slid significantly.
- Decision: classify as no_valid_simple_primitive / avoid_required / recovery_required_before_entry.

## Policy V1 direction
- Do not treat all recovery as `safe_stop`.
- Distinguish:
  - `safe_stop`: stable if robot stops in place.
  - `avoid_required`: terrain should not be entered; stopping on it may still fail.
  - `new_recovery_primitive_required`: needs controlled descent, crawl/kneel/slide, or backstep descent primitive.
