# ROS2-to-LL1 Clearance Authority

## Purpose

Verify the complete command path from a ROS 2 high-level reference to
the feature-gated LL1 swing-apex residual.

## Controller history

- `b9cab78`: feature-gated swing-apex residual
- `85adb97`: fail-closed finite/configuration safety
- `f6fb596`: realized foot-height authority evidence

## Runtime route

```text
ROS2 fixed reference publisher
  -> /tracer/mpc_reference
  -> tracer_ros2_mpc_ref_udp_sender.py
  -> UDP 127.0.0.1:50110, struct format <6d
  -> tracer_udp_to_ros1_mpc_ref.py
  -> ROS1 /tracer/mpc_reference
  -> GazeboA1ROS callback
  -> bounded swing-apex residual
Reference layout:

[counter, vx, yaw_rate, body_height, clearance, enable]
Conditions

Two independent canonical-reset trials were run.

Trial	Clearance	Expected residual
Neutral	0.045 m	0 mm
High	0.060 m	+15 mm

Each trial used:

one ROS 2 publisher;
one ROS 2 UDP sender;
one ROS 1 UDP receiver;
one LL1 controller;
20 Hz high-level reference publication;
five seconds of measured controller operation.
Results
Measurement	Neutral	High
ROS1 reference packets	101	101
Unique debug packets	62	62
Maximum swing shape	1.0	1.0
Maximum applied bump	0 mm	+15 mm
Raw-command error	0	0
Delta error	0	0
Bump relation error	0	0
Maximum stance bump	0	0
Bounded robot state	true	true
Conclusion

The ROS 2 clearance command was preserved across the UDP bridge and
consumed by the LL1 controller with the expected neutral-relative
mapping:

0.045 m -> 0 mm
0.060 m -> +15 mm

This closes target-level end-to-end clearance authority.

Actual foot-position realization was established separately in the
matched-phase A/B evidence from commit f6fb596.

This experiment does not establish improved terrain traversal or use
J7 itself as the ROS 2 source.

Safe final state

After the experiment:

ROS 2 publisher and sender were stopped;
the UDP 50110 receiver was stopped;
the LL1 controller was stopped;
no joint-command publisher remained;
the feature was restored to false;
Gazebo physics was paused.
