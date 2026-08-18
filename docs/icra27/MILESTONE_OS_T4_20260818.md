# ICRA27 TRACER-HL Milestone — OS-T4
Date: 2026-08-18

## Scope

This milestone freezes the current ICRA27 high-level planner study.

Low-level execution remains frozen:
- Quadruped-PyMPC backend
- exact lockstep evaluation: 100 x 0.002 s = 0.200 s / HL decision
- fixed swing clearance: 0.060 m
- learned HL action:
  [vx, yaw_rate, body_height]
- beta-conditioned observation:
  obs21 + beta3 = 24D

Current reward:
- tracer_cost_v4

Frozen working checkpoint:
- results/icra27/phase1a_beta_conditioned_v4_seed27027/checkpoint_update_0030.pt

No RAM/GMS/perception is included in this milestone.

---

## Closed Milestones

### OS-T4.6 — Traction semantics mechanism audit
CLOSED

Historical tracer_cost_v3 used binary physical-contact age.
Short contact losses inside an otherwise continuous planned stance
reset the contact-age gate and repeatedly restored the touchdown
grace period.

This caused low-friction traction-cost sign inversions.

The 40 ms touchdown gate itself was retained.

### OS-T4.7 — tracer_cost_v4 promotion
CLOSED

v4 uses latched planned-stance contact semantics:

- first actual touchdown starts the contact-age clock
- micro contact loss does not reset the clock
- planned swing terminates the stance episode
- traction cost remains active only while physical contact is present

Reward algebra and beta scalarization remain unchanged from v3.

### OS-T4.8c — Fresh v4 PPO training
PASS

Fresh training from scratch:
- 40 PPO updates
- 12 beta/terrain conditions per update
- no v3 warm start
- no M4 interventions
- no failures after update 19

### OS-T4.8d — Train-only checkpoint selection
PASS

Candidate checkpoints:
- u10
- u20
- u30
- u40

Semantic gate:

u10:
- Motion 8/9
- Stability 0/9
- Energy 6/9
- Low-friction traction 0/3

u20:
- Motion 9/9
- Stability 2/9
- Energy 6/9
- Low-friction traction 0/3

u30:
- Motion 9/9
- Stability 6/9
- Energy 9/9
- Low-friction traction 3/3
- success 36/36
- M4 0

u40:
- Motion 9/9
- Stability 5/9
- Energy 8/9
- Low-friction traction 3/3

Selected checkpoint:
- update 30

Selection was performed using TRAIN-only contexts.

### OS-T4.8e — Beta specialization audit
COMPUTE PASS

u30 target-cost behavior:

- Motion:
  better than balanced 9/9
  strict rank-1 8/9

- Stability:
  better than balanced 6/9
  strict rank-1 0/9

- Energy:
  better than balanced 9/9
  strict rank-1 8/9

Conclusion:
beta-conditioning is active, but objective disentanglement is partial.

### OS-T4.8f — Beta action geometry audit
COMPUTE PASS

Global action PCA:
- PC1 explained variance: 0.960866
- PC2: 0.036843
- PC3: 0.002292

PC1 direction:
[vx, yaw, height]
=
[+0.688278, -0.336382, -0.642744]

Mean delta cosine:
- Motion vs Stability: +0.790179
- Motion vs Energy:    -0.950419
- Stability vs Energy: -0.939892

Interpretation:

Beta-conditioned behavior is strongly organized along one dominant
coordinated action manifold.

Motion and Stability lie largely on the same side of this manifold,
while Energy lies approximately in the opposite direction.

### OS-T4.9b — Frozen-u30 primary held-out evaluation
PASS

- checkpoint fixed before held-out evaluation
- deterministic tanh(mu) policy
- tracer_cost_v4
- exact lockstep
- 116 / 116 successful episodes
- no checkpoint reselection after held-out results

### OS-T4.9c — Held-out per-context semantic audit
PASS

Primary evidence: 9 held-out rough-terrain realizations.

Motion:
- directional improvement: 9/9
- strict target-cost rank-1: 7/9

Stability:
- directional improvement: 5/9
- strict target-cost rank-1: 1/9

Energy:
- directional improvement: 8/9
- strict target-cost rank-1: 8/9

Interpretation:

Motion preference:
- strong transferable specialization

Energy preference:
- strong specialization with respect to designed C_E
- physical episode energy reduction is NOT yet established

Stability preference:
- partial directional specialization
- weak independent objective specialization
- likely constrained by the compact 3D meta-action authority

---

## Current Research Conclusion

The frozen planner

    pi_HL(a | s, beta)

does not ignore beta.

Preference changes produce systematic changes in the high-level
meta-command and meaningful objective tradeoffs that generalize to
held-out rough terrain, especially for Motion and Energy.

However, the current 3D meta-action space

    [vx, yaw_rate, body_height]

does not provide strongly disentangled Stability behavior.

Do NOT retrain or modify the reward solely because of this result.

---

## Next Experimental Gate

Before developing the Objective Selector, establish that preference
conditioning provides practical value relative to simpler baselines.

Compare:

B0 — Fixed meta-command
B1 — Frozen u30 with balanced beta
B2 — Frozen u30 with objective-conditioned beta

All comparisons must use:
- same frozen PyMPC low-level
- same transition semantics
- same terrain seeds / initial conditions
- same task
- paired evaluation

Metrics:

Task:
- success
- M4 intervention
- terminal/fall
- final goal error
- completion time

Motion:
- C_M
- mean vx
- progress rate

Stability:
- C_posture
- C_traction
- C_S
- roll/pitch RMS
- max abs roll/pitch

Energy:
- designed C_E
- physical mechanical energy
- energy / distance
- mechanical CoT if available

If B2 demonstrates meaningful objective-specific benefit without
sacrificing task feasibility, proceed to Objective Selector learning.

---

## Next Research Stage After Baseline Gate

1. Harden preferred-trajectory metric contract.
2. Extract preferred / dominated trajectory pairs.
3. Build terrain-conditioned preference dataset.
4. Train Objective Selector:
       context -> beta
5. Compare:
       Fixed
       Balanced beta
       Oracle beta
       Learned Objective-Selector beta

