# ICRA27 M3 — Frequency × Duty Characterization v0

## Scope

Flat MuJoCo / Go2, PyMPC + WBC.

Fixed:
- vx = 0.20 m/s
- yaw rate = 0
- body height = 0.30 m
- swing clearance = 0.06 m

Structural variables:
- gait frequency
- duty factor

## Main findings

### 1. Command authority

Frequency and duty-factor commands are propagated into the PyMPC gait
generator and produce corresponding planned contact schedules.

### 2. Physical contact realization

Physical contact duty generally follows the planned schedule in valid
rollouts, but contact realization depends on the joint frequency-duty
condition.

A simple independent box bound on frequency and duty factor is not a
sufficient description of locomotion viability.

### 3. Direct-start vs guarded-entry behavior

Observed operational classes include:

- robust candidate
- entry-sensitive
- transition-sensitive
- persistent problem

Therefore viability depends not only on the final (frequency, duty)
command but also on the entry state / gait phase / transition history.

### 4. Full-stance transition guard

Structural transition at planned full stance is useful, but is not a
universal safety guarantee.

Some commands that fail under direct startup survive guarded entry,
while other commands still fail after a valid full-stance commit.

### 5. Failure precursor

Corrected first-rollout-only post-commit support-count deficit:

- f=1.1, D=0.50, survived: 0.0107
- f=1.1, D=0.55, failed:   0.2399
- f=1.1, D=0.60, survived: 0.0775
- f=1.0, D=0.55, failed:   0.2449

In the failing f=1.1, D=0.55 rollout, substantial physical support
deficit appears while body roll is still moderate, followed shortly by
large roll divergence and non-foot ground contact.

This supports the interpretation that physical support mismatch is a
failure precursor. It does not yet establish a unique root cause.

### 6. Foothold observation

Failing cases show increasingly irregular physical touchdown locations
together with attitude divergence. However, isolated unusual touchdown
positions are also observed in surviving cases.

Therefore bad touchdown placement alone is not treated as a proven root
cause.

## Design implication

For the TRACER low-level interface:

1. project MetaGaitCommand into backend physical limits;
2. guard structural transitions;
3. do not equate a valid projected command with guaranteed locomotion
   viability;
4. retain post-transition physical-contact / stability monitoring as a
   later low-level safety mechanism.

## Status

Frequency × duty characterization slice: CLOSED for M3 v0.

Further investigation is intentionally deferred unless required by the
final operating-envelope study.
