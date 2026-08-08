# ICRA27 Progress

## M0 — Vanilla Quadruped-PyMPC baseline

- [x] Clean ICRA27 branch/worktree
- [x] Dedicated Conda environment
- [x] Quadruped-PyMPC pinned submodule
- [x] acados build and Tera code generation
- [x] MuJoCo Go2 spawn
- [x] PyMPC whole-body torque closed loop
- [x] Repeated locomotion baseline

## M1 — TRACER → PyMPC Low-Level Interface

- [x] Upstream command-path code audit
- [x] Fixed vx command authority
- [x] Yaw-rate authority
- [x] Body-height authority
- [x] Swing-clearance authority
- [x] Gait-period / step-frequency authority
- [x] Duty-factor authority

## M2 — MetaGaitCommand Adapter

- [x] MetaGaitCommand → PyMPC references
- [x] Safety bounds
- [x] Rate limiting
- [x] Gait timing synchronization
- [x] Safe dynamic command transitions — guarded full-stance transition PASS

## M3 — Action-Authority Characterization

- [~] Single-variable sweeps — preliminary 3-point sensitivity summary PASS; denser/repeatability pending
- [x] vx × yaw coupling — weak coupling over vx=0.1–0.3 m/s, yaw gain≈0.80–0.82
- [ ] Command vs internal reference logging
- [ ] Command vs measured response logging
- [ ] Sensitivity / authority table
- [ ] Remove or restrict ineffective/unsafe dimensions

## M4 — Closed TRACER Low-Level Box

- [x] Fixed MetaGaitCommand → torque
- [x] Runtime-changing MetaGaitCommand → torque
- [x] Clean shutdown/reset
- [x] Reproducible launcher
- [x] Meta-gait safety projection / admissibility
- [x] Continuous reference slew limiting
- [x] Full-stance atomic structural gait transition
- [x] Runtime physical-health monitor
- [x] UNSAFE latch → guarded known-nominal back-off
- [x] Canonical deterministic M4 evaluation
- [x] Explicit M4 ON/OFF paired ablation
- [x] Episode reset lifecycle / clean-state semantics
- [x] Reusable `PyMPCLowLevelRuntime`
- [x] Standalone nominal live validation
- [x] Standalone M4 intervention live validation
- [x] Closed low-level E2E acceptance

M4-v0 behavior is frozen as:

- NORMAL → pass-through
- WATCH → diagnostic only
- UNSAFE → latch/reject active high-level target and request
  guarded return to the known nominal operating point

This is an empirical runtime execution supervisor / command governor,
not a formally certified CLF/CBF safety controller. Recovery is not
guaranteed.

## M5 — ROS2 Interface

- [x] MetaGaitCommand ROS2 interface
- [x] ROS2 ↔ PyMPC process separation via localhost UDP
- [x] Versioned named-field command/telemetry protocol
- [x] ROS2 → PyMPC command adapter
- [x] PyMPC → ROS2 telemetry/output topics
- [x] ROS source freshness gate
- [x] UDP transport timeout → known nominal through the same closed runtime
- [x] Structural command → guarded full-stance atomic commit
- [x] Simulation-time and structural-commit-count observability
- [x] Canonical M3 structural transition cross-validated through ROS2/UDP

M5 frozen boundary:

- ROS2 command topic: `/tracer/meta_gait_cmd`
- command fields: `vx`, `yaw_rate`, `body_height`, `swing_clearance`,
  `gait_period`, `duty_factor`
- command UDP: `127.0.0.1:50510`
- telemetry UDP: `127.0.0.1:50511`
- ROS source stale timeout: 0.25 s
- PyMPC UDP stale timeout: 0.25 s
- stale transport falls back to the known nominal command through
  `PyMPCLowLevelRuntime` and `TransitionManager`; it does not bypass M4
- ROS/PyMPC Python ABI separation is intentional:
  ROS2 Humble uses Python 3.10 while the PyMPC/acados environment uses
  Python 3.12

Canonical structural live acceptance:

- nominal: `f=1.4 Hz`, `D=0.65`
- target: `f=1.1 Hz`, `D=0.50`
- ROS2/UDP target first applied at simulation `t=3.012 s`
- guarded structural commit at simulation `t=3.216 s`
- `UNSAFE=None`
- `override=False`
- `terminations=0`
- UDP bad packets: 0
- standalone hooks restored after run

The ROS2/UDP boundary therefore preserves the closed low-level runtime
semantics characterized in M2–M4. M5 is an interface boundary, not a
separate controller or formal safety layer.

## M6 — Minimal High-Level Reconnection

- [ ] Fixed high-level command
- [ ] Scheduled high-level command
- [ ] Minimal learned-policy reconnection
- [ ] Decide when to reconnect Objective Selector / RAM / GMS

<!-- M3_FREQUENCY_DUTY_V0 -->
### M3 frequency × duty characterization v0

- [x] planned → physical contact instrumentation
- [x] stability-aware first-rollout analysis
- [x] direct-start 3×3/local coupling characterization
- [x] guarded-entry paired viability characterization
- [x] contact/attitude/foothold failure-precursor trace
- [x] cross-reset time-aliasing contamination identified and excluded
- [x] frequency × duty M3 v0 slice closed
- [ ] final global operating envelope deferred until remaining action pairs

<!-- M3_HEIGHT_CLEARANCE_V0 -->
### M3 body height × swing clearance characterization v0

- [x] physical height / swing-clearance instrumentation
- [x] central-point smoke validation
- [x] 3×3 body-height × clearance characterization
- [x] 9/9 tested points viable
- [x] approximately linear diagonal authority
- [x] weak measured off-diagonal coupling in tested range
- [x] body height × swing clearance M3 v0 slice closed
- [x] targeted repeatability — tested protocol is effectively deterministic across seeds 0/1/2
- [ ] consolidated M3 operating envelope

<!-- M3_CONSOLIDATED_V0 -->
### M3 consolidated action-space characterization v0

- [x] vx / yaw locomotion authority
- [x] body-height / swing-clearance authority
- [x] body-height × clearance weak-coupling characterization
- [x] frequency × duty conditional-viability characterization
- [x] physical gait cadence verified from touchdown intervals
- [x] targeted deterministic repeatability
- [x] continuous action group identified: vx / yaw / height / clearance
- [x] structural action group identified: frequency / duty
- [x] M3 v0 characterization sufficient for M4 low-level-box closure
