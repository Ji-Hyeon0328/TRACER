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

- [ ] Fixed MetaGaitCommand → torque
- [ ] Runtime-changing MetaGaitCommand → torque
- [ ] Clean shutdown/reset
- [ ] Reproducible launcher

## M5 — ROS2 Interface

- [ ] MetaGaitCommand ROS2 interface
- [ ] ROS2 → PyMPC adapter
- [ ] Telemetry/output topics

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
