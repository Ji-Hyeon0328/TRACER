# TRACER High-level Checkpoint — 2026-06-27

## Branch

- `telecommunicating-structure`

## Current milestone

The current TRACER high-level stack has a working learned/meta-gait runtime path over the Gazebo A1 QP-MPC bridge.

Current active path:

```text
terrain / policy entry
  -> objective selector baseline
  -> low-alpha learned beta blend
  -> learned stack v3 / GMS
  -> theta MLP UDP policy
  -> theta decoder / mapper
  -> MPC reference [vx, yaw_rate, body_height, swing_clearance, enable]
  -> ROS2 to UDP to ROS1 bridge
  -> A1 QP-MPC Gazebo controller
Current safety / adaptation observation path:
online proprio / odom sanitizer
  -> online RAM UDP client
  -> RAM gate monitor
  -> GMS input / high-level debug
  -> RAM recovery shadow gate
  -> rollout recorder metrics
Confirmed tags
tracer-online-feature-sanitize-v0
tracer-online-feature-sanitize-beta-blend-robust-pass
tracer-ram-recovery-shadow-gate-report-v0
tracer-ram-recovery-shadow-runtime-v0
tracer-ram-recovery-shadow-recorded-v0
tracer-ram-recovery-shadow-recorded-robust-pass
tracer-rollout-debug-freshness-summary-v0
1. Objective Selector / beta learning

Status:

✅ simulation rollout dataset scaffold
✅ trajectory metric extraction scaffold
✅ preference / ranking construction scaffold
✅ objective selector dataset v1/v2
✅ objective selector baseline model
✅ beta shadow debug path
✅ beta low-alpha blend active path
✅ beta blend robust30 validation
🟨 IRL / preference learning baseline
🟨 beta predictor baseline
🟨 beta-conditioned policy validation
⬜ full beta replacement validation

Notes:

The learned beta is not fully replacing the rule beta yet. The current deployed path uses a low-alpha blend between rule beta and learned beta.

2. Meta-gait policy

Status:

✅ input / output interface
✅ theta decoder / mapper
✅ theta action space v0
✅ theta warm-start dataset
✅ BC-style warm-start MLP
✅ theta MLP training / evaluation
✅ runtime learned theta inference via UDP
✅ learned stack v3 + GMS + beta blend runtime validation
🟨 beta, rho, sigma conditioned policy validation
⬜ context c_t conditioned policy validation
⬜ PPO / SAC RL fine-tuning
⬜ true RL meta-gait policy

Notes:

The current theta policy is a BC / warm-start MLP. It is connected to runtime and validated through the bridge, but it is not yet an RL-fine-tuned policy.

3. RAM / recovery gate / adaptation

Status:

✅ RAM window dataset scaffold
✅ RAM intervention predictor v1
✅ RAM shadow comparison utilities
✅ online RAM UDP client
✅ online odom / proprio feature sanitizer
✅ post-sanitize beta blend robust30 pass
✅ RAM recovery shadow gate offline report
✅ RAM recovery shadow gate runtime debug
✅ rollout recorder shadow metric capture
✅ recorded shadow robust pass
✅ debug freshness summary metrics
🟨 RAM recovery active candidate design
⬜ RAM recovery active gate validation
⬜ actual recovery primitive design
⬜ recovery primitive execution in low-level / controller path
⬜ RAM-conditioned impedance / residual adaptation

Notes:

The current RAM recovery logic is shadow-only. It records whether recovery would be triggered, but it does not modify the active command path.

The next candidate should be an active gate, not a complex recovery primitive yet.

Recommended first active candidates:
Candidate A: active hold
  vx = 0
  enable = 1
  keep or slightly modify body height / clearance

Candidate B: conservative recovery command
  vx very small
  body height slightly higher
  clearance higher
  slower gait period
  higher duty factor
Backstep / crawl / controlled-slide primitives should be treated as later research primitives.

4. Context Encoder

Status:

⬜ camera / depth / elevation encoder
⬜ proprio-history encoder
⬜ terrain context c_t
⬜ contrastive / predictive training
⬜ real-world deployable encoder

Notes:

The high-level stack already has places where c_t can later be connected, but the actual context encoder has not started yet.

5. Low-level Controller

Status:

✅ ROS1 / ROS2 bridge scaffold
✅ MPC reference UDP bridge
✅ proprio / odom bridge
✅ online feature sanitizer
✅ Gazebo A1 QP-MPC controller connection
🟨 high-level command overlay validation
⬜ TRACER-native MPC
⬜ TRACER-native WBC
⬜ impedance residual controller
⬜ RAM-conditioned residual / adaptive impedance
⬜ recovery primitive execution layer
⬜ Isaac / ROS2 native low-level port

Notes:

The current low-level path uses an open-source A1 QP-MPC controller through a bridge. TRACER-native MPC/WBC/residual control is still future work.

6. Evaluation / Ablation

Status:

✅ flat / rough / slope smoke validation
✅ learned stack v3 GMS-only robust30
✅ beta blend robust30
✅ post-sanitize robust30
✅ RAM recovery shadow recorded robust pass
🟨 terrain-wise quantitative evaluation
⬜ larger terrain set evaluation
⬜ ablation: rule-only vs learned GMS
⬜ ablation: no beta vs beta shadow vs beta blend vs full beta
⬜ ablation: no RAM vs RAM shadow vs RAM active
⬜ ablation: bridge low-level vs TRACER-native low-level
⬜ real-world transfer evaluation
Next priority
RAM recovery active candidate v0
Keep the first active candidate minimal and reversible
Do not start with crawl / backstep / sliding primitive
Start with shadow-tested trigger plus conservative command override
Validate on flat / rough / slope with robust30 before expanding terrain set

