from tracer_core.common.ref_command import RefCommand


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class ThetaToRefMapper(object):
    """
    First-stage TRACER mapper.

    theta -> RefCommand
    RefCommand -> Gazebo /joy is handled by the adapter in tracer_cmd_bridge.py.
    """

    def __init__(self, vx_min=0.0, vx_max=0.05, yaw_max=0.05, accel_limit=0.005):
        self.vx_min = vx_min
        self.vx_max = vx_max
        self.yaw_max = yaw_max
        self.accel_limit = accel_limit
        self.prev_vx = 0.0

    def map(self, theta):
        emergency_stop = bool(theta.get("emergency_stop", False))
        if emergency_stop:
            self.prev_vx = 0.0
            return RefCommand(
                vx=0.0,
                vy=0.0,
                yaw_rate=0.0,
                height_rate=0.0,
                walk=False,
                emergency_stop=True,
            )

        target_vx = float(theta.get("target_vx_axis", 0.0))
        target_yaw = float(theta.get("target_yaw_axis", 0.0))
        walk = bool(theta.get("walk", False))

        target_vx = clamp(target_vx, self.vx_min, self.vx_max)
        target_yaw = clamp(target_yaw, -self.yaw_max, self.yaw_max)

        dv = clamp(target_vx - self.prev_vx, -self.accel_limit, self.accel_limit)
        vx = self.prev_vx + dv
        self.prev_vx = vx

        return RefCommand(
            vx=vx,
            vy=0.0,
            yaw_rate=target_yaw,
            height_rate=0.0,
            walk=walk,
            emergency_stop=False,
        )
