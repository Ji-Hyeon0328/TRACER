# TRACER Phase-J10b Downslope Force-6 Policy v1

This creates a conservative tabular policy variant that forces downslope to use action 6.

- source policy: `models/phase_j/j2_meta_action_tabular_policy_v0.json`
- output policy: `models/phase_j/j2_meta_action_tabular_policy_v1_downslope_force6.json`
- paired bank: `configs/phase_j/meta_action_bank_v1_downslope_conservative.json`

## Rationale

- J10 shadow showed downslope/action 6 passed the J7 gate almost always.
- J10 also showed downslope/action 7 was rejected by the guard.
- Therefore, before active downslope opening, use a controlled policy where downslope maps to conservative action 6.

## Before downslope policy

```json
{
  "action_probs": [
    {
      "action_id": 0,
      "action_name": "nominal_cruise",
      "count": 0,
      "prob": 0.00010051261433309881
    },
    {
      "action_id": 1,
      "action_name": "fast_motion",
      "count": 0,
      "prob": 0.00010051261433309881
    },
    {
      "action_id": 2,
      "action_name": "energy_saver",
      "count": 0,
      "prob": 0.00010051261433309881
    },
    {
      "action_id": 3,
      "action_name": "rough_high_clearance",
      "count": 0,
      "prob": 0.00010051261433309881
    },
    {
      "action_id": 4,
      "action_name": "rough_stability",
      "count": 0,
      "prob": 0.00010051261433309881
    },
    {
      "action_id": 5,
      "action_name": "upslope_push",
      "count": 0,
      "prob": 0.00010051261433309881
    },
    {
      "action_id": 6,
      "action_name": "downslope_stable",
      "count": 449,
      "prob": 0.9027037893255604
    },
    {
      "action_id": 7,
      "action_name": "lateral_recovery_soft",
      "count": 28,
      "prob": 0.05638757664086843
    },
    {
      "action_id": 8,
      "action_name": "goal_hold",
      "count": 20,
      "prob": 0.04030555834757262
    }
  ],
  "rows": 497,
  "source": "high_conf",
  "top_action_id": 6,
  "top_action_name": "downslope_stable"
}
```

## After downslope policy

```json
{
  "action_id": 6,
  "action_name": "downslope_stable",
  "reason": "J10 shadow showed conservative action 6 passes J7 guard while action 7 is rejected in downslope.",
  "source": "manual_conservative_downslope_override_j10b"
}
```

## Safe interpretation

- This is still for shadow validation first.
- It does not change rough behavior.
- It prepares a controlled J11 active candidate where downslope projected references are conservative and guard-compatible.
