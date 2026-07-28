# J7 Interleaved ABBA Absolute Swing-Foot Authority

## Purpose

Verify that a clearance residual selected by the J7 guard reaches the low-level swing controller and produces a positive absolute swing-foot response without restarting the controller or resetting the robot between conditions.

## Design

One canonical standing reset, one LL1 controller, one J7 guard, one ROS2-to-UDP sender, one UDP-to-ROS1 receiver, and one continuous Gazebo rollout were used.

A1 and A2 used projected 0.060 m, which J7 rejected to empirical 0.045 m. B1 and B2 used projected 0.050 m, which J7 accepted as a +0.005 m LL1 residual. The ABBA effect was mean(B1, B2) minus mean(A1, A2).

## Result

Across 12 matched apex cells and all four legs:

- Median native effect: -3.042848 mm.
- Median applied-bump effect: +4.608000 mm.
- Median final-target effect: +1.957152 mm.
- Median absolute-actual effect: +1.376650 mm.
- Positive final-target cells: 83.3%.
- Positive absolute-actual cells: 66.7%.
- Median realized/target ratio: 0.703395.

The J7-selected clearance residual reached LL1 and produced a positive absolute swing-foot response over the bracketing empirical fallback.

This supports physical command authority. It does not establish terrain benefit, stability benefit, energy benefit, robust generalization, Objective Selector readiness, or RL/meta-policy performance improvement.
