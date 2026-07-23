# TRACER Phase-K4 Invalid Candidate Study Audit v0

Phase-K4 flat micro-action candidate study was invalid as a candidate-performance comparison.

## Root cause

The K4 flat micro-action override node crashed at startup:

```text
AttributeError: can't set attribute 'context'
The node attempted to assign self.context, but rclpy.node.Node already owns a read-only context property.

Effect

Because the override node died:

K4 override theta was not published
-> J4 did not produce projected references
-> J7 reported missing_projected_ref
-> J7 output failsafe_hold
-> robot stayed near the start and all active candidate modes failed to reach the goal

Therefore, K4 results should be interpreted as an infrastructure audit, not as evidence that the micro-action candidates are bad.

Fix

K4B renames the internal state variable from self.context to self.terrain_context.
