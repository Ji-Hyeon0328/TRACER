# Phase-B LL1 Bank–J7 Semantic Collision Resolution v1

## Finding

The frozen J7 gate interprets `theta[0]` as a legacy gate action token and
rejects token value `8`. The new LL1-aligned bank legitimately defines bank
action IDs `0..8`.

The v0 mapper placed the true bank action ID in `theta[0]`. Consequently,
bank action 8 would be rejected by J7 even though its command deltas satisfy
the numerical guard.

## Resolution

The v1 mapper separates gate control from policy identity:

- `theta[0]`: constant active J7 gate token `1`
- `theta[1]`: true bank action ID `0..8`
- `theta[8]`: hold override

The actual action ID also remains available in the mapper debug payload and
the `/tracer/hl_action_index` input.

## Scope

This change does not modify the frozen J7 implementation and does not make
a Gazebo feasibility or performance claim. It only removes the semantic ID
collision before runtime collection.
