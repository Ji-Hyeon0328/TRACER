import math
from collections import deque


JOY_CMD_VELX_MAX = 0.6  # from A1Params.h


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mean(vals):
    if not vals:
        return 0.0
    return sum(vals) / float(len(vals))


def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    var = sum((v - m) * (v - m) for v in vals) / float(len(vals))
    return math.sqrt(var)


class RobustAdaptationModule(object):
    """
    M7: RAM-style mismatch history module.

    This is still rule/proxy-based, not learned yet.

    Inputs:
        state
        last_target_vx_axis

    Outputs:
        rho_v_inst
        rho_v_mean
        sigma_v
        v_cmd
        v_meas
    """

    def __init__(self, history_len=40):
        self.history_len = int(history_len)
        self.rho_v_hist = deque(maxlen=self.history_len)

    def update(self, state, last_target_vx_axis):
        lin_vel = state.get("linear_velocity", (0.0, 0.0, 0.0))

        v_meas_xy = math.sqrt(
            lin_vel[0] * lin_vel[0] +
            lin_vel[1] * lin_vel[1]
        )

        v_cmd = abs(last_target_vx_axis) * JOY_CMD_VELX_MAX

        if v_cmd < 1e-6:
            rho_v_inst = 0.0
        else:
            rho_v_inst = abs(v_cmd - v_meas_xy) / max(v_cmd, 1e-6)
            rho_v_inst = clamp(rho_v_inst, 0.0, 1.0)

        self.rho_v_hist.append(rho_v_inst)

        hist_vals = list(self.rho_v_hist)
        rho_v_mean = clamp(mean(hist_vals), 0.0, 1.0)
        sigma_v = clamp(std(hist_vals), 0.0, 1.0)

        return {
            "v_cmd": v_cmd,
            "v_meas": v_meas_xy,
            "rho_v_inst": rho_v_inst,
            "rho_v_mean": rho_v_mean,
            "sigma_v": sigma_v,
        }
