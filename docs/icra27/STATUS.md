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

- [x] Fixed high-level command
- [x] Scheduled high-level command
- [x] Minimal learned-policy reconnection
- [ ] Decide when to reconnect Objective Selector / RAM / GMS

### M6.1 fixed high-level reconnection

- [x] Separate `tracer_highlevel` ROS2 package
- [x] High-level node is independent of PyMPC / M4 / TransitionManager
- [x] Publishes `/tracer/meta_gait_cmd` as six values:
  `vx`, `yaw_rate`, `body_height`, `swing_clearance`,
  `gait_period`, `duty_factor`
- [x] Fixed command ROS wire contract validated
- [x] One high-level publisher and one frozen-M5 bridge subscriber
- [x] Live high-level → ROS2 → UDP → frozen low-level → MuJoCo E2E

Canonical M6.1 live acceptance:

- fixed high-level command:
  `vx=0.12`, `yaw_rate=0.0`, `body_height=0.30`,
  `swing_clearance=0.06`, `gait_period=1/1.4`, `duty_factor=0.65`
- PyMPC received the high-level command from simulation start
- final command sequence: 507
- UDP bad packets: 0
- `UNSAFE=None`
- `override=False`
- `terminations=0`
- standalone hooks restored after run

M6.1 changes only the command producer above the frozen M5 boundary.
The PyMPC low-level runtime and M4 execution semantics remain unchanged.

### M6.2 scheduled high-level reconnection

- [x] Continuous high-level schedule through the frozen M5 boundary
- [x] Structural high-level schedule through the frozen M5 boundary
- [x] Continuous command changes observed during live locomotion
- [x] Structural frequency / duty change committed exactly once
- [x] Single high-level publisher requirement verified
- [x] High-level process cleanup verified after the live run
- [x] Frozen low-level runtime left unchanged

Canonical M6.2 continuous live acceptance:

- scheduled `vx` sequence:
  `0.12 -> 0.16 -> 0.08 -> 0.12 m/s`
- `yaw_rate=0.0`, `body_height=0.30`,
  `swing_clearance=0.06`, `f=1.4 Hz`, `D=0.65` held fixed
- the full requested sequence reached the PyMPC runtime
- UDP bad packets: 0
- `UNSAFE=None`
- `override=False`
- `terminations=0`
- standalone hooks restored after run

Canonical M6.2 structural live acceptance:

- fixed:
  `vx=0.20`, `yaw_rate=0.0`, `body_height=0.30`,
  `swing_clearance=0.06`
- nominal structural command:
  `f=1.4 Hz`, `D=0.65`
- target structural command:
  `f=1.1 Hz`, `D=0.60`
- requested structural states changed exactly once
- structural commit occurred exactly once at simulation
  `t=3.572 s`
- UDP bad packets: 0
- `UNSAFE=None`
- `override=False`
- `terminations=0`
- standalone hooks restored after run
- no high-level producer process remained after cleanup

A contaminated structural trial exposed that multiple concurrent
publishers on `/tracer/meta_gait_cmd` can alternate incompatible
high-level requests. M6.2 therefore treats one active high-level
publisher as an execution precondition.

A lower-margin trial using `vx=0.12 m/s` with the structural target
`f=1.1 Hz`, `D=0.50` triggered M4 intervention in the tested entry
condition. It is retained as a stress observation, not as the
canonical M6.2 acceptance case. This is consistent with the M3
finding that structural-command viability depends on entry state,
gait phase, and transition history.

M6.2 therefore establishes dynamic high-level command authority across
the ROS2/UDP boundary without changing the frozen M5 low-level
execution semantics.


### M6.3 learned-policy reconnection

- [x] Supervised learned surrogate artifact generated from the existing
  rule-based high-level teacher
- [x] Direct Torch model reload / inference validated
- [x] Torch inference isolated behind the existing UDP policy boundary
- [x] Torch-free / `tracer_core`-free ROS2 learned command producer
- [x] Dynamic learned ROS command wire validated with `fast` and
  `conservative` gait modes
- [x] Learned continuous outputs separated from uncharacterized legacy
  structural outputs
- [x] Learned command propagated through frozen M5 into live PyMPC
- [x] UDP bad packets: 0
- [x] `UNSAFE=None`
- [x] `override=False`
- [x] `terminations=0`
- [x] standalone hooks restored after run
- [x] learned-policy / ROS / PyMPC background-process cleanup verified

M6.3 is an integration and execution-boundary validation, not a final
learned-policy performance result. The current model is a supervised
surrogate trained from 42 samples generated by the existing rule-based
teacher; it is not the final TRACER RL planner and does not establish
policy generalization.

The canonical M6.3 learned inference produced approximately:

- learned continuous command:
  `vx=0.2778`, `yaw_rate=0.0`, `body_height=0.2960`,
  `swing_clearance=0.0320`
- raw learned structural prediction:
  `gait_period=0.3188 s`, `duty_factor=0.5290`
- executed structural request:
  `gait_period=1/1.4 s`, `duty_factor=0.65`

The structural fields were intentionally held at the characterized PyMPC
nominal reference for the canonical E2E acceptance. This separates the
learned inference / command plumbing test from compatibility of legacy
teacher structural outputs with the new PyMPC operating region.

Canonical M6.3 live E2E acceptance:

- one active learned high-level ROS2 publisher
- learned command received by the frozen M5 UDP runtime
- accepted command:
  `vx=0.278`, `yaw_rate=0.000`, `body_height=0.296`,
  `swing_clearance=0.032`, `gait_period=0.714`, `duty_factor=0.650`
- structural commits: none
- M4 safety / termination events: none
- final command sequence: 18054
- UDP bad packets: 0
- `UNSAFE=None`
- `override=False`
- `terminations=0`
- standalone hooks restored after run
- policy UDP `50310` and runner UDP `50510` released after cleanup;
  frozen M5 bridge telemetry port `50511` remained active as expected

One interface limitation is intentionally deferred beyond M6.3: the
current frozen M5 command wire contains only the six executable meta-gait
fields and does not carry the learned policy's `enable` output. Therefore,
`recovery` / `no_valid` disable semantics must be defined explicitly before
GMS or other mode-selection logic is allowed to exercise those modes live.

M6.3 therefore establishes that an independently hosted learned inference
artifact can act as a real high-level command source through ROS2 and the
frozen M5 low-level execution boundary. It does not yet reconnect the
Objective Selector, RAM, GMS, full terrain context, goal conditioning, or
the final high-level RL planner.



<!-- M7_HIGHLEVEL_RL_V0 -->
## M7 — High-Level RL Meta-Action Planner

Scope:

- frozen M4 / M5 / PyMPC low-level execution semantics
- given / oracle terrain context
- no RAM
- no Objective Selector
- no GMS
- one active high-level command producer

### M7.0 architecture / legacy audit

- [x] Existing high-level / RL asset inventory
- [x] Classify legacy RL code as REUSE / ADAPT / PARK
- [x] Continuous vs structural action separation retained from M3
- [x] M7-v0 structural action held at characterized nominal
- [x] Observation contract frozen
- [x] Action range / normalization contract frozen
- [x] Reward-v0 component contract frozen
- [x] High-level decision frequency frozen
- [x] M7 environment / state-tap boundary frozen


M7.0 frozen decisions:

- policy action: normalized continuous `[-1,1]^4`
- zero normalized action maps to characterized nominal
  `[vx=0.20, yaw=0.0, h=0.30, clr=0.06]`
- physical action envelope:
  `vx=[0.00,0.40] m/s`,
  `yaw=[-0.40,0.40] rad/s`,
  `h=[0.24,0.32] m`,
  `clr=[0.03,0.09] m`
- structural M7-v0:
  `gait_period=1/1.4 s`, `duty_factor=0.65`
- observation:
  `oracle context[K] + goal[4] + base response[6] +
  applied command[4] + previous action[4]`
- observation dimension: `K + 18`
- high-level decision rate: `5 Hz` (`0.2 s`)
- M5 transport may repeat the latest action at its existing
  higher publish rate; policy inference remains 5 Hz
- reward-v0 is fixed-weight and componentized; no Objective
  Selector / RAM / GMS is active
- M4 UNSAFE / override is exposed as an explicit negative
  reward component
- M7 physical-state instrumentation is a read-only wrapper/tap;
  frozen M4/M5 execution semantics are not modified

### M7.1 minimal closed RL environment

- [x] Read-only MuJoCo physical-state tap

  - acceptance: UDP `50512`, 20 Hz read-only state stream
  - smoke: 20 packets, monotonic seq/sim-time
  - state phase: `controller_input_pre_env_step`
  - frozen M5 result: no UNSAFE / override / termination
  - standalone hooks restored after run
- [x] Gymnasium reset / step semantics

  - real subprocess-backed PyMPC episode on `reset()`
  - observation: `K+18`; smoke used `K=3 -> 21`
  - point goal: reset pose + `0.5 m` forward
  - policy action interval observed at approximately `0.20–0.21 s`
  - oracle context propagated into observation
  - reward uses measured decision dt
  - M4 UNSAFE/override observed live and penalized in reward
  - same-seed fresh-process reset max observation difference: `0.0`
  - JSONL episode logs created
  - runner / transport / UDP cleanup passed
- [x] Requested vs applied action observability

  - isolated M7 training ports:
    command `50610`, telemetry `50611`, state `50612`
  - scripted actions: nominal / faster / slow-turn / geometry
  - M5 `command_seq` matched each logical policy action
  - requested and guarded/applied actions separately observable
  - physical state returned for each transition
  - transition sim-time approximately `0.20–0.24 s`
  - no bad packets / UNSAFE / override / termination
- [x] M4 intervention exposed in reward / info
- [x] Oracle terrain context input
- [x] Scripted/random-action closed-loop smoke through frozen M5
- [x] Deterministic reset smoke
- [x] Episode logging / cleanup validation

### M7.2 RL training smoke

- [ ] Select PPO / SAC implementation
- [ ] Continuous 4D policy training smoke
- [ ] Reward improves beyond random-policy baseline
- [ ] Policy checkpoint save / reload
- [ ] Deterministic evaluation rollout

### M7.3 learned-policy M5 E2E

- [ ] Learned policy inference through ROS2
- [ ] Learned MetaGaitCommand through frozen M5
- [ ] Requested vs applied action trace
- [ ] M4 intervention rate reported
- [ ] Learned vs heuristic/fixed baseline comparison

### Deferred beyond M7-v0

- [ ] Restricted structural action bank
- [ ] Learned gait-period / duty-factor selection
- [ ] RAM reconnection
- [ ] Objective Selector reconnection
- [ ] GMS reconnection


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
