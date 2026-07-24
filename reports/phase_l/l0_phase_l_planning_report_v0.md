# Phase-L0: Data-Backed Meta-Action Selector Planning

## 1. Purpose

Phase-L starts after the closure of Phase-K.

The goal of Phase-L is to move away from manual meta-action tuning and toward a
data-backed meta-action selector / RL meta-planner prototype while preserving the
current safe empirical controller path.

Phase-L does NOT immediately deploy a learned active policy.

The initial focus is:

1. preserve the empirical/default runtime as the safe baseline,
2. reuse Phase-J/K outcomes as negative and near-miss supervision,
3. expand the candidate/action dataset beyond flat terrain,
4. build a conservative selector in shadow mode,
5. evaluate whether candidate actions can consistently outperform the empirical baseline,
6. promote an active learned policy only after sufficient rollout evidence exists.

---

## 2. Current Safe Runtime Baseline

The current deployable runtime is:

```text
terrain/context
    |
    v
D7 Objective Selector beta
    |
    v
D6 beta-conditioned gated selector
    |
    v
empirical/default MPC reference routing
    |
    v
A1-QP-MPC
    |
    v
Gazebo rollout
The Phase-J/K guarded meta-action route remains available only as a
shadow/debug/promotion scaffold.

candidate meta-action
    |
    v
J4 projection
    |
    v
J7 guarded reference gate
    |
    +---- rejected / inactive ----> empirical reference
    |
    +---- future promoted action -> active reference

At Phase-L start, no learned or manually tuned candidate is promoted as an
active performance policy.

3. Phase-J/K Conclusion

Phase-J demonstrated that the guarded routing scaffold itself works.

However, repeated active evaluations did not establish a manually tuned
meta-action candidate that consistently outperformed the empirical baseline.

Phase-K therefore concluded:

default runtime action: empirical,
deploy active performance policy: false,
active modes: none,
flat_slow03: shadow-only,
remaining tested manual candidates: rejected,
manual micro-action tuning: stop.

This means that continuing to hand-tune isolated action candidates is unlikely
to be the best use of additional rollout budget.

Phase-L instead treats previous candidate outcomes as supervision for a
data-backed decision process.

4. Main Research Question

The primary Phase-L question is:

Given terrain context, objective weights, mismatch/uncertainty, locomotion
state, segment transition state, and a candidate meta-action, when should
TRACER retain the empirical baseline and when is the candidate likely to
improve rollout performance?

The first Phase-L model is therefore not assumed to directly generate a
globally optimal continuous meta-action.

Instead, it first learns conservative candidate evaluation.

Conceptually:

state/context
+ objective beta
+ mismatch/uncertainty
+ candidate action
        |
        v
data-backed candidate evaluator
        |
        +---- reject -----------> empirical active
        |
        +---- shadow_only ------> empirical active
        |                         candidate evaluated in shadow
        |
        +---- positive evidence -> still shadow initially
                                  repeated evaluation
                                  guarded promotion test
                                  J7 active gate only later
5. Why Candidate Evaluation Comes Before Active RL Deployment

The current dataset does not contain a sufficiently established positive active
policy class.

Most Phase-J/K candidates are either:

rejected,
invalid,
near-miss,
or shadow-only.

Therefore a direct policy learner trained to imitate the current "best action"
could incorrectly learn artifacts from a sparse candidate set.

Phase-L should first learn:

when a candidate is unsafe or clearly inferior,
when a candidate is worth further shadow evaluation,
which state/action regions lack sufficient evidence,
eventually, which candidates show statistically repeatable improvement over baseline.

This creates a safer path toward a learned meta-action policy.

6. Phase-L Decision Labels

The dataset should preserve the following semantic labels.

empirical

No candidate has sufficient evidence to replace the empirical/default action.

Runtime behavior:

active = empirical
reject

The candidate is invalid, unsafe, unstable, or clearly inferior.

Runtime behavior:

active = empirical
candidate = rejected
shadow_only

The candidate is plausible or near-baseline but does not have sufficient
evidence for active deployment.

Runtime behavior:

active = empirical
candidate = shadow evaluation
candidate_positive_later

Reserved for future candidates that repeatedly outperform the empirical
baseline under comparable state/context conditions.

This label must not imply immediate deployment.

Runtime behavior during early Phase-L remains:

active = empirical
candidate = extended shadow evaluation

Only after separate promotion criteria are satisfied may the candidate be
considered for J7 guarded active routing.

7. Dataset Design

The Phase-L dataset should separate input features from rollout outcomes.

A. Situation Features
context_label
next_context
segment_id
segment_progress

x
y
abs_y

context_at_action_start
action_start_x
action_start_y

These variables describe where the robot is and the current terrain transition.

B. Objective and Adaptation Features
beta_motion
beta_stability
beta_energy

ram_rho_summary
sigma_uncertainty

The initial implementation may use available RAM proxies if full RAM features
are not yet consistently logged.

The schema should allow future replacement with richer rho/sigma features.

C. Baseline Action Features
baseline_vx
baseline_yaw
baseline_body_h
baseline_clearance
baseline_enable

These are necessary because the candidate should be interpreted relative to the
empirical action available in the same state.

D. Candidate Action Features
candidate_mode
candidate_action_id

candidate_vx
candidate_yaw
candidate_body_h
candidate_clearance
candidate_enable

candidate_vx_scale
candidate_vx_delta
candidate_yaw_gain
candidate_body_h_delta
candidate_clearance_delta

Long-term learning should prefer the relative candidate representation:

candidate_action - baseline_action

rather than memorizing candidate names.

E. Routing / Guard Features
j7_accept_rate
j7_reject_rate
failsafe_hold_rate

These fields capture whether the proposed action actually reached the active
routing path during evaluation.

F. Outcome and Supervision
goal_reached
final_x

max_abs_y
mean_abs_y

delta_progress
time_to_goal

risk_score
termination_reason

decision_label

The exact risk score definition should be versioned and should not be inferred
from a single metric alone.

8. Provenance Fields

Every dataset row or summarized candidate outcome should retain provenance.

source_phase
source_rollout_id
source_log
world_name
seed_or_trial_id
candidate_config

This is important because Phase-J/K contain:

valid baseline trials,
valid candidate trials,
invalid candidate studies,
override-node failures,
failsafe-hold-only runs.

Invalid runs must not accidentally become valid performance supervision.

9. Initial Phase-L Model Scope

The first learned selector should run in shadow mode only.

Recommended initial modeling tasks:

L1

Define the expanded Phase-L dataset schema and map existing K6/K7/K8 artifacts
into that schema.

L2

Build a dataset builder that merges:

Phase-K candidate outcome data,
baseline rollout data,
corrected segment metrics,
future broader terrain candidate trials.
L3

Train a conservative candidate evaluator.

Possible initial targets:

reject
shadow_only
empirical

candidate_positive_later should only become a meaningful supervised class
after positive evidence has been collected.

L4

Run selector predictions in shadow mode against new rollouts.

L5

Generate broader candidate/action data for:

flat
rough
upslope
downslope
goal transition
terrain transitions
L6

Evaluate repeatability against the empirical baseline.

Only sufficiently repeatable improvements should enter an explicit promotion
study.

10. Safety / Deployment Rule

Throughout the initial Phase-L work:

active runtime = empirical/default

The learned/data-backed selector is:

shadow-only

No candidate is routed actively solely because the learned selector predicts it
to be beneficial.

Future active promotion requires:

offline evidence,
repeated rollout evidence,
comparable baseline trials,
explicit promotion criteria,
J7 guarded routing.
11. Relationship to Long-Term TRACER Architecture

Phase-L is an intermediate step toward the intended TRACER structure:

terrain/context
+ RAM mismatch/uncertainty
        |
        v
Objective Selector beta
        |
        v
learned meta-gait planner / selector
        |
        v
Theta / meta-action
        |
        v
Theta decoder / Theta-to-ref mapper
        |
        v
low-level MPC/WBC or A1-QP-MPC
        |
        v
robot rollout

The initial Phase-L candidate evaluator should therefore be designed so that it
can later evolve into:

a learned discrete meta-action selector,
contextual bandit policy,
offline RL policy,
or continuous residual/meta-action policy.

The current empirical action remains the safety anchor while this transition is
validated.

12. Phase-L0 Decision

Phase-L begins with the following decisions:

stop manual micro-action tuning,
preserve empirical/default as the active runtime,
preserve J7 as guarded routing infrastructure,
treat Phase-J/K failures and near-misses as useful supervision,
build the dataset before expanding learned policy complexity,
begin learned selection in shadow mode only,
expand candidate collection across terrain contexts,
require repeated evidence before any future active promotion.

The immediate next step is Phase-L1:

inspect the actual K6/K7/K8 data structures and define the versioned
Phase-L selector dataset schema without inventing unavailable fields.
