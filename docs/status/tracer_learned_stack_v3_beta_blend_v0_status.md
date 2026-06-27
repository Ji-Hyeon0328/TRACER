# TRACER Learned Stack v3 Beta Blend v0 Status

## Summary

This milestone validates the first limited activation of learned objective beta in the TRACER high-level stack.

The active learned components are:

- learned GMS label
- learned objective beta low-alpha blend

The following components remain inactive:

- full learned beta replacement
- learned RAM hard gate
- learned objective score command selector
- RAM-triggered recovery

## Safety Scope

Beta blend v0 is intentionally conservative.

Safety constraints:

- low-alpha beta blend, default alpha = 0.10
- terrain-specific max delta limit
- GMS-only safety path maintained
- rule fallback remains available
- RAM recovery probability is logged but not used as an active hard gate

## Current Active Stack

```text
Runtime policy path:

policy_json / terrain semantic
  -> runtime objective selector / rule beta
  -> learned stack v3 UDP query
  -> learned GMS label, gated by safety checks
  -> learned beta shadow
  -> low-alpha limited beta blend
  -> meta-gait theta policy
  -> decoder / mapper
  -> MPC reference
Validation Results
Short Smoke

Beta blend v0 short smoke passed on:

flat_normal
rough_mid
slope_5deg

Observed beta deltas remained small.

terrain	delta_mean	vx_mean	success
flat_normal	m=-0.015, s=+0.006, e=+0.009	0.2800	True
rough_mid	m=-0.004, s=+0.007, e=-0.003	0.0478	True
slope_5deg	m≈0.000, s≈-0.001, e≈0.000	0.0410	True
Robust30 Evaluation

Robust evaluation was run with:

terrains: flat_normal, rough_mid, slope_5deg
repeats: 3
duration: 30 seconds per episode
sample rate: 5 Hz
beta blend alpha: 0.10

Result: PASS

terrain	n	success	vx_mean	body_h	clearance	gate_override
flat_normal	3	1.000	0.2800±0.0000	0.2950±0.0000	0.0301±0.0000	0.0000±0.0000
rough_mid	3	1.000	0.0391±0.0002	0.3497±0.0014	0.1087±0.0008	0.7444±0.0113
slope_5deg	3	1.000	0.0366±0.0005	0.3517±0.0044	0.1123±0.0013	0.7467±0.0109
Interpretation

The learned beta blend did not destabilize the existing GMS-only active path.

The main effect of beta blend v0 is intentionally small. It slightly adjusts the objective beta before meta-gait theta generation while keeping the overall command behavior close to the validated rule/GMS-safe behavior.

The high ram_recovery_needed_mean values observed during online rollout are not treated as failures in this milestone because RAM-triggered recovery is not active. This remains a known online RAM scalar calibration issue.

Current Milestone State
TRACER learned stack v3:

Active:
  - learned GMS label
  - learned objective beta low-alpha blend

Validated:
  - beta shadow
  - beta blend short smoke
  - beta blend robust30

Status:
  PASS

Not active:
  - full beta replacement
  - learned RAM hard gate
  - learned objective score command selector
  - RAM-triggered recovery
Relevant Tags
tracer-learned-stack-v3-gms-only-deploy
tracer-learned-stack-v3-gms-only-recovery-guard
tracer-learned-stack-v3-robust-report-pass-flag
tracer-learned-stack-v3-beta-shadow-raw-objective-fix
tracer-learned-stack-v3-beta-shadow-report
tracer-learned-stack-v3-beta-blend-v0
tracer-learned-stack-v3-beta-blend-report
tracer-learned-stack-v3-beta-blend-robust-eval-utils
Recommended Next Steps
Keep alpha = 0.10 as the default active beta blend setting.
Run alpha sweep only in controlled evaluation, not as default deploy.
Add beta blend alpha sweep report for alpha = 0.05, 0.10, 0.15.
Continue treating learned RAM as shadow-only until online scalar calibration is fixed.
Do not enable RAM-triggered recovery before solving the proprio/RAM magnitude issue.
