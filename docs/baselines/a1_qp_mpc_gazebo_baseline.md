# A1-QP-MPC Gazebo Baseline

## Purpose

This baseline provides the first verified model-based low-level locomotion backbone for TRACER.

TRACER does not claim a new low-level MPC/WBC solver. Instead, this baseline uses a public A1 model-based controller and connects TRACER later through an external reference interface.

## Source

Original repository:

- ShuoYangRobotics/A1-QP-MPC-Controller

Original commit:

- 79c91302b5855cf245a852aefa7162449bf4ac9e

Local baseline branch during setup:

- tracer/a1_qp_mpc_baseline

## Local patches

Two local patches are included.

1. Build fix

Disabled optional test targets `test_rotation` and `test_bezier` in `src/a1_cpp/CMakeLists.txt`.

Reason:

- The optional test target failed to build due to an OsqpEigen / OSQP header mismatch involving `auxil.h`.
- The controller executable `gazebo_a1_ctrl` does not require these tests.

2. Gazebo standstill handoff fix

Added XY reference locking in standstill mode in `src/a1_cpp/src/GazeboA1ROS.cpp`.

Reason:

- During Gazebo handoff, `root_pos_d.xy` can remain near the world origin.
- The patch locks desired XY position to the current base position when `movement_mode == 0`.

Patch file:

- `patches/a1_qp_mpc_gazebo_baseline.patch`

## Tested status

Confirmed:

- Gazebo A1 world launches.
- A1-QP-MPC controller builds and launches.
- Neutral standing handoff works.
- Walking mode toggle works.
- Forward walking with `/joy axes[5] = 0.03` was observed.

## Command interface

The low-level controller subscribes to `/joy`.

Mapping:

```text
buttons[0] -> walking mode toggle
axes[5]    -> forward velocity command
axes[2]    -> lateral velocity command
axes[0]    -> yaw rate command
axes[1]    -> body height velocity command
axes[7]    -> pitch rate command
axes[6]    -> roll rate command

TRACER high-level output
    β, ρ, σ, context
        ↓
Θ-Decoder
        ↓
Θ-to-Reference Mapper
        ↓
synthetic /joy publisher
        ↓
A1-QP-MPC low-level controller

---

## 5. `.gitignore` 추가

```bash
cat > .gitignore <<'EOF'
# ROS / Gazebo / catkin artifacts
build/
devel/
install/
logs/
*.bag
*.log
.ros/

# Python
__pycache__/
*.pyc

# Runtime / crash outputs
*.core
core
