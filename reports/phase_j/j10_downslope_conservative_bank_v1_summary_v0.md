# TRACER Phase-J10 Conservative Downslope Bank v1

This creates a conservative variant of the Phase-J meta-action bank for downslope opening.

- source bank: `configs/phase_j/meta_action_bank_v0.json`
- output bank: `configs/phase_j/meta_action_bank_v1_downslope_conservative.json`
- target action: `downslope_stable` / action id `6`

## Intended change

| parameter | new value |
|---|---:|
| vx_scale | 0.88 |
| yaw_gain_scale / yaw_gain | 0.90 |
| body_h_delta | -0.004 |
| clearance_delta | 0.004 |

## Safe interpretation

- This bank is for shadow validation first, not immediate active control.
- The intent is to reduce the downslope velocity change enough to pass the J7 guard.
- Rough actions are intentionally unchanged and should remain rejected/protected for now.

## Before

```json
{
  "description": "Reduce speed and increase stability margin for downslope traversal.",
  "id": 6,
  "intended_contexts": [
    "downslope"
  ],
  "name": "downslope_stable",
  "theta": {
    "body_h_delta": -0.006,
    "clearance_delta": 0.004,
    "energy_bias": -0.1,
    "hold_override": false,
    "stability_bias": 0.3,
    "vx_delta": 0.0,
    "vx_scale": 0.78,
    "yaw_gain_scale": 0.8
  }
}
```

## After

```json
{
  "description": "Reduce speed and increase stability margin for downslope traversal.",
  "id": 6,
  "intended_contexts": [
    "downslope"
  ],
  "name": "downslope_stable",
  "theta": {
    "body_h_delta": -0.004,
    "clearance_delta": 0.004,
    "energy_bias": -0.1,
    "hold_override": false,
    "stability_bias": 0.3,
    "vx_delta": 0.0,
    "vx_scale": 0.88,
    "yaw_gain_scale": 0.9
  }
}
```
