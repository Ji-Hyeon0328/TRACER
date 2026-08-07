# ICRA27 M3 — Consolidated Meta-Gait Action Characterization v0

## Objective

Characterize which MetaGaitCommand dimensions provide effective
low-level authority through Quadruped-PyMPC, how strongly they couple,
and whether they can be treated as independent high-level actions.

The tested MetaGaitCommand dimensions are:

- forward velocity
- yaw rate
- body height
- swing clearance
- gait frequency
- duty factor

The results in this document are local to the tested flat-terrain
MuJoCo / Go2 / PyMPC regime. They are not global safety guarantees.

---

## 1. Forward velocity

Tested range:

- 0.0 to 0.4 m/s

Preliminary 1-D characterization:

- measured gain: approximately 0.895
- intercept: approximately +0.033 m/s
- R2: approximately 0.99995

Interpretation:

Forward velocity has strong physical authority, with a small positive
low-speed bias.

Recommended treatment:

- continuous high-level action
- projected to the tested controller range
- rate limited during runtime transitions

---

## 2. Yaw rate

Tested range:

- -0.4 to +0.4 rad/s

Preliminary 1-D characterization:

- measured gain: approximately 0.845
- intercept: approximately -0.003 rad/s
- R2: approximately 0.999997

The vx × yaw coupling test showed no strong forward-speed dependence in
the tested regime, although yaw response is consistently attenuated.

Recommended treatment:

- continuous high-level action
- projected and rate limited
- do not assume unity physical gain

---

## 3. Body height

Tested range:

- 0.24 to 0.32 m

Body-height gain across three fixed-clearance slices:

- approximately 1.030 to 1.032

The measured body height contains an approximately +0.02 to +0.03 m
systematic offset relative to the requested reference.

Interpretation:

The offset is a tracking/reference bias rather than lack of command
authority.

Recommended treatment:

- continuous high-level action
- projected and slew limited
- physical height should be monitored separately from the requested
  reference

---

## 4. Swing clearance

Tested range:

- 0.03 to 0.09 m

Measured clearance gain across three fixed-body-height slices:

- approximately 0.964 to 0.968

Physical clearance therefore tracks the requested value with only small
attenuation.

Recommended treatment:

- effective high-level action
- retain projection
- coordinate structural application with gait phase where needed

---

## 5. Body height × swing clearance coupling

A 3×3 characterization over:

- body height = {0.24, 0.28, 0.32} m
- clearance = {0.03, 0.06, 0.09} m

showed:

- 9 / 9 tested combinations viable
- no termination
- max observed |roll| approximately 0.67 deg
- max observed |pitch| approximately 2.28 deg

Off-diagonal effects were small.

Measured body-height change across the full clearance sweep:

- approximately 2.1 to 2.2 mm

Measured-clearance change across the full body-height sweep:

- approximately 0.3 to 0.5 mm

Interpretation:

Within the tested nominal regime, body height and swing clearance behave
as approximately independent action dimensions.

---

## 6. Gait frequency

Tested command range:

- 1.0 to 2.0 Hz

Internal PyMPC gait-generator authority is approximately unity.

Physical contact cadence was also checked using touchdown intervals.

For the robust representative condition:

- command: f = 1.1 Hz, D = 0.60
- aggregate physical touchdown frequency median: 1.0953 Hz
- command error: -0.0047 Hz

All four legs showed regular cycle frequencies close to the commanded
cadence.

For the lower-margin surviving condition:

- command: f = 1.1 Hz, D = 0.50
- FR / RL / RR cadence remained near the commanded frequency
- FL contained additional irregular touchdown edges

Therefore a pooled touchdown frequency alone can hide per-leg contact
irregularity.

Interpretation:

Gait frequency propagates to physical contact cadence, but frequency
realization should be evaluated together with contact regularity.

Recommended treatment:

- structural high-level action
- not treated as an independently safe scalar
- transition guard and post-transition monitoring required

---

## 7. Duty factor

Tested range:

- 0.50 to 0.80

The planned contact schedule follows the requested duty factor and
physical contact duty generally tracks it on valid trajectories.

However, frequency × duty characterization showed non-monotonic
viability.

Observed operational classes included:

- robust candidate
- entry-sensitive
- transition-sensitive
- persistent problem under the tested protocols

Direct-start and guarded-entry outcomes differed for multiple points.

Interpretation:

Duty factor cannot be assigned an independent static safe bound that is
valid for all tested entry states and transition histories.

Recommended treatment:

- structural high-level action
- coupled with gait frequency
- guarded transition
- physical contact / stability monitoring after transition

---

## 8. Frequency × duty conditional viability

The tested results reject a simple Cartesian safe region of the form:

    frequency_range × duty_range

as a sufficient safety model.

A more appropriate representation is conditional viability:

    V(f, D, entry_state, gait_phase, transition_history)

A planned full-stance atomic commit improves some transitions but is not
a universal stability guarantee.

Selected failed transitions showed physical support deficit before
major attitude divergence. This supports support mismatch as an
instability precursor, but does not establish a unique root cause.

---

## 9. Recommended high-level action structure

The six effective dimensions should not be treated identically.

Continuous locomotion/reference actions:

    a_cont = [vx, yaw_rate, body_height, swing_clearance]

Structural gait actions:

    a_struct = [gait_frequency, duty_factor]

The continuous group exhibits strong direct authority and relatively
weak coupling in the tested nominal regime.

The structural group changes hybrid contact timing and exhibits
state-, phase-, and transition-history-dependent viability.

---

## 10. Implication for the closed TRACER low-level box

The low-level interface should therefore contain at least:

1. MetaGaitCommand projection into tested backend limits
2. slew limiting for continuous commands
3. phase-aware handling of structural commands
4. synchronization of frequency / duty / swing timing internals
5. post-transition physical-contact and attitude monitoring
6. rejection or recovery behavior when the resulting gait becomes
   dynamically unsafe

A projected command is not equivalent to a dynamically safe command.

---

## M3 conclusion

The effective TRACER low-level command space is:

    [vx, yaw_rate, body_height, swing_clearance,
     gait_frequency, duty_factor]

but it is not a homogeneous six-dimensional independent Box action
space.

Body height and swing clearance behave approximately independently in
the tested nominal regime. Forward velocity and yaw rate provide usable
continuous locomotion authority. Gait frequency and duty factor provide
real command authority but require coupled, transition-aware treatment.

M3 v0 characterization is complete enough to proceed to M4 low-level
box closure.
