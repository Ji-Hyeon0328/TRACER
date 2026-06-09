import math


def quat_to_roll_pitch_yaw(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class RuleBasedHighLevelPlanner(object):
    """
    Minimal closed-loop TRACER high-level shell.

    stable posture -> walk slowly
    medium roll/pitch -> conservative command
    dangerous roll/pitch or low base height -> emergency stop
    """

    def __init__(self,
                 nominal_vx_axis=0.03,
                 conservative_vx_axis=0.01,
                 max_roll_pitch_soft=0.25,
                 max_roll_pitch_hard=0.45,
                 min_base_height=0.18):
        self.nominal_vx_axis = nominal_vx_axis
        self.conservative_vx_axis = conservative_vx_axis
        self.max_roll_pitch_soft = max_roll_pitch_soft
        self.max_roll_pitch_hard = max_roll_pitch_hard
        self.min_base_height = min_base_height

    def compute_theta(self, state):
        q = state.get("orientation", None)
        z = float(state.get("base_height", 0.0))

        if q is None:
            return {
                "walk": False,
                "target_vx_axis": 0.0,
                "target_yaw_axis": 0.0,
                "emergency_stop": False,
            }

        roll, pitch, _ = quat_to_roll_pitch_yaw(q[0], q[1], q[2], q[3])
        rp_abs = max(abs(roll), abs(pitch))

        if z < self.min_base_height or rp_abs > self.max_roll_pitch_hard:
            return {
                "walk": False,
                "target_vx_axis": 0.0,
                "target_yaw_axis": 0.0,
                "emergency_stop": True,
            }

        if rp_abs > self.max_roll_pitch_soft:
            return {
                "walk": True,
                "target_vx_axis": self.conservative_vx_axis,
                "target_yaw_axis": 0.0,
                "emergency_stop": False,
            }

        return {
            "walk": True,
            "target_vx_axis": self.nominal_vx_axis,
            "target_yaw_axis": 0.0,
            "emergency_stop": False,
        }
