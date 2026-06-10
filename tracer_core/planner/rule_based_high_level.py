import math
from collections import deque


JOY_CMD_VELX_MAX = 0.6  # from A1Params.h


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


class RuleBasedHighLevelPlanner(object):
    """
    M5: mismatch-history-based TRACER shell.

    This is still a rule-based placeholder, but now it contains a RAM-like
    history buffer:

        rho_v_inst : instantaneous command tracking mismatch
        rho_v_mean : recent mismatch mean
        sigma_v    : recent mismatch variation

    Later, this block can be replaced by a learned RAM that outputs rho/sigma.
    """

    def __init__(self,
                 nominal_vx_axis=0.12,
                 conservative_vx_axis=0.05,
                 max_roll_pitch_soft=0.25,
                 max_roll_pitch_hard=0.45,
                 min_base_height=0.18,
                 mismatch_soft=0.45,
                 mismatch_hard=0.75,
                 sigma_soft=0.18,
                 sigma_hard=0.30,
                 history_len=40):
        self.nominal_vx_axis = nominal_vx_axis
        self.conservative_vx_axis = conservative_vx_axis

        self.max_roll_pitch_soft = max_roll_pitch_soft
        self.max_roll_pitch_hard = max_roll_pitch_hard
        self.min_base_height = min_base_height

        self.mismatch_soft = mismatch_soft
        self.mismatch_hard = mismatch_hard
        self.sigma_soft = sigma_soft
        self.sigma_hard = sigma_hard

        self.history_len = int(history_len)
        self.rho_v_hist = deque(maxlen=self.history_len)

        # The mismatch at time t is measured against the command issued recently.
        self.last_target_vx_axis = self.nominal_vx_axis

    def compute_theta(self, state):
        q = state.get("orientation", None)
        z = float(state.get("base_height", 0.0))

        if q is None:
            return self._theta(
                mode="waiting",
                walk=False,
                target_vx_axis=0.0,
                emergency_stop=False,
                beta_v=0.0,
                beta_s=1.0,
                beta_e=0.0,
                rho=0.0,
                sigma=1.0,
                roll=0.0,
                pitch=0.0,
                v_cmd=0.0,
                v_meas=0.0,
                rho_v_inst=0.0,
                rho_v_mean=0.0,
                sigma_v=0.0,
            )

        roll, pitch, _ = quat_to_roll_pitch_yaw(q[0], q[1], q[2], q[3])
        rp_abs = max(abs(roll), abs(pitch))

        lin_vel = state.get("linear_velocity", (0.0, 0.0, 0.0))
        v_meas_xy = math.sqrt(lin_vel[0] * lin_vel[0] + lin_vel[1] * lin_vel[1])

        v_cmd = abs(self.last_target_vx_axis) * JOY_CMD_VELX_MAX

        if v_cmd < 1e-6:
            rho_v_inst = 0.0
        else:
            rho_v_inst = abs(v_cmd - v_meas_xy) / max(v_cmd, 1e-6)
            rho_v_inst = clamp(rho_v_inst, 0.0, 1.0)

        self.rho_v_hist.append(rho_v_inst)
        hist_vals = list(self.rho_v_hist)
        rho_v_mean = clamp(mean(hist_vals), 0.0, 1.0)
        sigma_v = clamp(std(hist_vals), 0.0, 1.0)

        rho_posture = clamp(rp_abs / self.max_roll_pitch_hard, 0.0, 1.0)
        sigma_posture = clamp(
            (rp_abs - self.max_roll_pitch_soft) /
            max(1e-6, self.max_roll_pitch_hard - self.max_roll_pitch_soft),
            0.0,
            1.0,
        )

        rho = max(rho_posture, rho_v_mean)
        sigma = max(sigma_posture, sigma_v)

        # Hard safety first.
        if z < self.min_base_height or rp_abs > self.max_roll_pitch_hard:
            theta = self._theta(
                mode="recovery",
                walk=False,
                target_vx_axis=0.0,
                emergency_stop=True,
                beta_v=0.0,
                beta_s=1.0,
                beta_e=0.0,
                rho=rho,
                sigma=sigma,
                roll=roll,
                pitch=pitch,
                v_cmd=v_cmd,
                v_meas=v_meas_xy,
                rho_v_inst=rho_v_inst,
                rho_v_mean=rho_v_mean,
                sigma_v=sigma_v,
            )
            self.last_target_vx_axis = 0.0
            return theta

        # Posture-conservative mode.
        if rp_abs > self.max_roll_pitch_soft:
            target = self.conservative_vx_axis
            theta = self._theta(
                mode="conservative_posture",
                walk=True,
                target_vx_axis=target,
                emergency_stop=False,
                beta_v=0.2,
                beta_s=0.7,
                beta_e=0.1,
                rho=rho,
                sigma=sigma,
                roll=roll,
                pitch=pitch,
                v_cmd=v_cmd,
                v_meas=v_meas_xy,
                rho_v_inst=rho_v_inst,
                rho_v_mean=rho_v_mean,
                sigma_v=sigma_v,
            )
            self.last_target_vx_axis = target
            return theta

        # History-based mismatch conservative mode.
        if rho_v_mean > self.mismatch_hard or sigma_v > self.sigma_hard:
            beta_v = 0.25
            beta_s = 0.45
            beta_e = 0.30
            target = self.beta_to_vx_axis(beta_v, beta_s, beta_e)
            theta = self._theta(
                mode="conservative_mismatch",
                walk=True,
                target_vx_axis=target,
                emergency_stop=False,
                beta_v=beta_v,
                beta_s=beta_s,
                beta_e=beta_e,
                rho=rho,
                sigma=sigma,
                roll=roll,
                pitch=pitch,
                v_cmd=v_cmd,
                v_meas=v_meas_xy,
                rho_v_inst=rho_v_inst,
                rho_v_mean=rho_v_mean,
                sigma_v=sigma_v,
            )
            self.last_target_vx_axis = target
            return theta

        if rho_v_mean > self.mismatch_soft or sigma_v > self.sigma_soft:
            beta_v = 0.45
            beta_s = 0.35
            beta_e = 0.20
            target = self.beta_to_vx_axis(beta_v, beta_s, beta_e)
            theta = self._theta(
                mode="cautious_mismatch",
                walk=True,
                target_vx_axis=target,
                emergency_stop=False,
                beta_v=beta_v,
                beta_s=beta_s,
                beta_e=beta_e,
                rho=rho,
                sigma=sigma,
                roll=roll,
                pitch=pitch,
                v_cmd=v_cmd,
                v_meas=v_meas_xy,
                rho_v_inst=rho_v_inst,
                rho_v_mean=rho_v_mean,
                sigma_v=sigma_v,
            )
            self.last_target_vx_axis = target
            return theta

        # Nominal mode.
        beta_v = 0.7
        beta_s = 0.2
        beta_e = 0.1
        target = self.beta_to_vx_axis(beta_v, beta_s, beta_e)
        theta = self._theta(
            mode="nominal",
            walk=True,
            target_vx_axis=target,
            emergency_stop=False,
            beta_v=beta_v,
            beta_s=beta_s,
            beta_e=beta_e,
            rho=rho,
            sigma=sigma,
            roll=roll,
            pitch=pitch,
            v_cmd=v_cmd,
            v_meas=v_meas_xy,
            rho_v_inst=rho_v_inst,
            rho_v_mean=rho_v_mean,
            sigma_v=sigma_v,
        )
        self.last_target_vx_axis = target
        return theta

    def beta_to_vx_axis(self, beta_v, beta_s, beta_e):
        """
        M6: beta-conditioned reference modulation.

        beta_v: progress / velocity preference
        beta_s: stability preference
        beta_e: energy-smoothness preference

        This keeps the output continuous instead of switching between
        discrete velocity commands only by mode.
        """
        energy_vx_axis = 0.5 * (self.nominal_vx_axis + self.conservative_vx_axis)

        vx = (
            beta_v * self.nominal_vx_axis +
            beta_s * self.conservative_vx_axis +
            beta_e * energy_vx_axis
        )

        return clamp(vx, 0.0, self.nominal_vx_axis)

    def _theta(self, mode, walk, target_vx_axis, emergency_stop,
               beta_v, beta_s, beta_e, rho, sigma, roll, pitch,
               v_cmd, v_meas, rho_v_inst, rho_v_mean, sigma_v):
        return {
            "mode": mode,
            "walk": walk,
            "target_vx_axis": target_vx_axis,
            "target_yaw_axis": 0.0,
            "emergency_stop": emergency_stop,
            "beta_v": beta_v,
            "beta_s": beta_s,
            "beta_e": beta_e,
            "rho": rho,
            "sigma": sigma,
            "roll": roll,
            "pitch": pitch,
            "v_cmd": v_cmd,
            "v_meas": v_meas,
            "rho_v_inst": rho_v_inst,
            "rho_v_mean": rho_v_mean,
            "sigma_v": sigma_v,

            # Backward-compatible field.
            # Existing bridge/analyzer can keep reading "rho_v".
            "rho_v": rho_v_mean,
        }
