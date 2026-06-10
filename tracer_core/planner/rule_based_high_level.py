import math


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


class RuleBasedHighLevelPlanner(object):
    """
    Minimal mismatch-aware TRACER high-level shell.

    This is still a rule-based placeholder, but now it contains the first
    TRACER-style feedback signal:

        rho_v = command tracking mismatch

    Later, rho_v will be replaced or augmented by the learned RAM output.
    """

    def __init__(self,
                 nominal_vx_axis=0.12,
                 conservative_vx_axis=0.05,
                 max_roll_pitch_soft=0.25,
                 max_roll_pitch_hard=0.45,
                 min_base_height=0.18,
                 mismatch_soft=0.65,
                 mismatch_hard=0.90):
        self.nominal_vx_axis = nominal_vx_axis
        self.conservative_vx_axis = conservative_vx_axis
        self.max_roll_pitch_soft = max_roll_pitch_soft
        self.max_roll_pitch_hard = max_roll_pitch_hard
        self.min_base_height = min_base_height
        self.mismatch_soft = mismatch_soft
        self.mismatch_hard = mismatch_hard

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
                rho_v=0.0,
            )

        roll, pitch, _ = quat_to_roll_pitch_yaw(q[0], q[1], q[2], q[3])
        rp_abs = max(abs(roll), abs(pitch))

        lin_vel = state.get("linear_velocity", (0.0, 0.0, 0.0))
        v_meas_xy = math.sqrt(lin_vel[0] * lin_vel[0] + lin_vel[1] * lin_vel[1])

        v_cmd = self.nominal_vx_axis * JOY_CMD_VELX_MAX
        rho_v = abs(v_cmd - v_meas_xy) / max(abs(v_cmd), 1e-6)
        rho_v = clamp(rho_v, 0.0, 1.0)

        rho_posture = min(1.0, rp_abs / self.max_roll_pitch_hard)
        rho = max(rho_posture, rho_v)

        sigma = min(1.0, max(0.0, (rp_abs - self.max_roll_pitch_soft) /
                             max(1e-6, self.max_roll_pitch_hard - self.max_roll_pitch_soft)))

        # Hard safety first.
        if z < self.min_base_height or rp_abs > self.max_roll_pitch_hard:
            return self._theta(
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
                rho_v=rho_v,
            )

        # Posture-conservative mode.
        if rp_abs > self.max_roll_pitch_soft:
            return self._theta(
                mode="conservative_posture",
                walk=True,
                target_vx_axis=self.conservative_vx_axis,
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
                rho_v=rho_v,
            )

        # Mismatch-conservative mode.
        # If command tracking is poor, reduce command but do not stop.
        if rho_v > self.mismatch_hard:
            return self._theta(
                mode="conservative_mismatch",
                walk=True,
                target_vx_axis=self.conservative_vx_axis,
                emergency_stop=False,
                beta_v=0.25,
                beta_s=0.45,
                beta_e=0.30,
                rho=rho,
                sigma=sigma,
                roll=roll,
                pitch=pitch,
                v_cmd=v_cmd,
                v_meas=v_meas_xy,
                rho_v=rho_v,
            )

        if rho_v > self.mismatch_soft:
            target = 0.5 * (self.nominal_vx_axis + self.conservative_vx_axis)
            return self._theta(
                mode="cautious_mismatch",
                walk=True,
                target_vx_axis=target,
                emergency_stop=False,
                beta_v=0.45,
                beta_s=0.35,
                beta_e=0.20,
                rho=rho,
                sigma=sigma,
                roll=roll,
                pitch=pitch,
                v_cmd=v_cmd,
                v_meas=v_meas_xy,
                rho_v=rho_v,
            )

        # Nominal mode.
        return self._theta(
            mode="nominal",
            walk=True,
            target_vx_axis=self.nominal_vx_axis,
            emergency_stop=False,
            beta_v=0.7,
            beta_s=0.2,
            beta_e=0.1,
            rho=rho,
            sigma=sigma,
            roll=roll,
            pitch=pitch,
            v_cmd=v_cmd,
            v_meas=v_meas_xy,
            rho_v=rho_v,
        )

    def _theta(self, mode, walk, target_vx_axis, emergency_stop,
               beta_v, beta_s, beta_e, rho, sigma, roll, pitch,
               v_cmd, v_meas, rho_v):
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
            "rho_v": rho_v,
        }
