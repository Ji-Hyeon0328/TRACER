# TRACER Phase-E3 Preference Fallback Deployment Profile v0

## Purpose

This profile converts the Phase-E3 bootstrap preference fallback evidence into a simple deployment rule.

The learned scorer itself is not a true IRL Objective Selector. It is a weak preference-supervised fallback/deployment scorer trained from E2 filtered preference pairs.

## Deployment rule

```text
if TRACER_RESET_Y_OFFSET < -0.15:
    use negative lateral fallback
    learned vx/yaw/body_h/clearance/enable = 0/0/0/1/0
else:
    use nominal full learned profile
    learned vx/yaw/body_h/clearance/enable = 1/1/0/1/0
Validation smoke results
condition	selected profile	success	goal	out_lane	final_y	max_abs_y	mean_abs_y	hold_drift	moving_accept_rate
reset_y=-0.30	clearance-only fallback	1	1	0	0.038	0.174	0.075	0.073	0.993432
reset_y=+0.30	full learned	1	1	0	-0.366	0.421	0.236	0.158	0.992222
Interpretation

The deployment profile correctly switches learned dimensions based on the lateral reset condition.

For negative lateral mismatch, the profile selects clearance-only fallback and preserves stable goal reaching.

For positive lateral offset, the profile selects full learned deployment. The rollout succeeds, but the lateral path quality is worse than the earlier p030 full-learned standalone run. This suggests the profile is valid as a smoke-tested fallback deployment wrapper, but more repeated trials are needed before claiming robust performance improvement.

Safe claim

Phase-E3 provides a smoke-tested runtime deployment profile that switches from full learned control to a clearance-only fallback under negative lateral mismatch. This is not yet a true IRL Objective Selector or learned RAM module.
