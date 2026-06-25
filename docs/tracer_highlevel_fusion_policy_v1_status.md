# TRACER High-Level Fusion Policy V1 Status

## Summary

The high-level Objective Selector and fusion-policy pipeline has been validated offline and through the ROS2 reference publisher.

Current pipeline:

1. Terrain/style mission logs are converted into trajectory features and preference pairs.
2. Objective IRL predicts terrain-conditioned objective weights:
   - beta_motion
   - beta_stability
   - beta_energy
   - recovery probability
3. High-level selector/fusion table combines:
   - objective recovery probability
   - RAM-like online risk
   - sigma/risk estimates
4. Fusion policy V1 exports terrain-level decisions into:
   - mode
   - semantic class
   - suggested style
   - risk
   - controller reference command

## Validated ROS2 output

Example terrain:

- terrain: sponge_firm_downslope_5deg_forward
- mode: locomotion
- semantic: validated_locomotion
- style: sponge_tall_10_c080
- risk: approximately 0.171

Published command:

```text
[counter, vx, yaw, body_height, clearance, enable]
[0.0, 0.1, 0.0, 0.335, 0.08, 1.0]
Interpretation

This validates the first end-to-end high-level path:
terrain/context
  -> objective beta / recovery probability
  -> fused terrain decision
  -> meta-style command
  -> controller-needed reference
  -> /tracer/mpc_reference
This is an initial form of the Theta-to-Reference mapping. The next step is to replace style-level command lookup with a richer Meta-gait plan theta and a Theta-Decoder / Theta-to-Ref Mapper.
