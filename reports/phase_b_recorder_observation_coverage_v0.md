# Phase-B Recorder Observation Coverage v0

## Current `phase_b_episode.csv` columns

The current Phase-B recorder stores only high-level reference, relative goal, and odometry-level signals:

- relative goal state: `rel_x`, `rel_y`, `rel_yaw`, `rel_dist`
- MPC reference: `mpc_vx`, `mpc_yaw_rate`, `mpc_body_height`, `mpc_swing_clearance`
- odometry: `odom_x`, `odom_y`, `odom_z`, `odom_yaw`, `odom_vx`, `odom_vy`

## Missing signals for invalid gait detection

The current recorder does not store:

- 12 joint positions
- 12 joint velocities
- 4 foot contact states
- base roll/pitch
- per-leg contact/motion statistics

## Consequence

The current logs cannot directly detect visually invalid gait patterns such as:

- RF/LH folded or inactive
- LF/RH dominant crawling/dragging
- diagonal-pair collapse
- contact asymmetry
- joint-limit folded posture

## Current workaround

`tracer_label_phase_b_suspicious_sponge_gait_v0.py` provides weak labels using only summary-level metrics such as:

- low reach but low final distance
- low progress
- stable final position without reaching goal threshold

These labels are useful for filtering suspicious targets but should not be treated as direct invalid-gait labels.

## Decision

Recorder should be extended before training a real invalid-gait detector.

Required future recorder fields:

- `q_*`: 12 joint positions
- `dq_*`: 12 joint velocities
- `contact_*`: 4 contact flags/probabilities
- `base_roll`
- `base_pitch`
- optionally `base_z`, although current `odom_z` is already recorded
