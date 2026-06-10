import math

from tracer_core.adaptation.robust_adaptation_module import RobustAdaptationModule


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
    M7: TRACER planner with separated RAM-style adaptation module.

    RAM:
        state + last command -> rho/sigma mismatch statistics

    Planner:
        posture + RAM output -> beta/mode/ref command
    """

    def __init__(self,
                 nominal_vx_axis=0.12,
                 conservative_vx_axis=0.05,
                 max_roll_pitch_soft=0.25,
                 max_roll_pitch_hard=0.45,
                 min_base_height=0.18,
                 mismatch_soft=0.50,
                 mismatch_hard=0.80,
                 sigma_soft=0.18,
                 sigma_hard=0.35,
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

        self.ram = RobustAdaptationModule(history_len=history_len)
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
                ram_out={
                    "v_cmd": 0.0,
                    "v_meas": 0.0,
                    "rho_v_inst": 0.0,
                    "rho_v_mean": 0.0,
                    "sigma_v": 0.0,
                },
            )

        roll, pitch, _ = quat_to_roll_pitch_yaw(q[0], q[1], q[2], q[3])
        rp_abs = max(abs(roll), abs(pitch))

        ram_out = self.ram.update(state, self.last_target_vx_axis)

        rho_posture = clamp(rp_abs / self.max_roll_pitch_hard, 0.0, 1.0)
        sigma_posture = clamp(
            (rp_abs - self.max_roll_pitch_soft) /
            max(1e-6, self.max_roll_pitch_hard - self.max_roll_pitch_soft),
            0.0,
            1.0,
        )

        rho_v_mean = ram_out["rho_v_mean"]
        sigma_v = ram_out["sigma_v"]

        rho = max(rho_posture, rho_v_mean)
        sigma = max(sigma_posture, sigma_v)

        if z < self.min_base_height or rp_abs > self.max_roll_pitch_hard:
            beta_v, beta_s, beta_e = 0.0, 1.0, 0.0
            target = 0.0
            theta = self._theta(
                mode="recovery",
                walk=False,
                target_vx_axis=target,
                emergency_stop=True,
                beta_v=beta_v,
                beta_s=beta_s,
                beta_e=beta_e,
                rho=rho,
                sigma=sigma,
                roll=roll,
                pitch=pitch,
                ram_out=ram_out,
            )
            self.last_target_vx_axis = target
            return theta

        if rp_abs > self.max_roll_pitch_soft:
            beta_v, beta_s, beta_e = 0.2, 0.7, 0.1
            target = self.beta_to_vx_axis(beta_v, beta_s, beta_e)
            theta = self._theta(
                mode="conservative_posture",
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
                ram_out=ram_out,
            )
            self.last_target_vx_axis = target
            return theta

        if rho_v_mean > self.mismatch_hard or sigma_v > self.sigma_hard:
            beta_v, beta_s, beta_e = 0.25, 0.45, 0.30
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
                ram_out=ram_out,
            )
            self.last_target_vx_axis = target
            return theta

        if rho_v_mean > self.mismatch_soft or sigma_v > self.sigma_soft:
            beta_v, beta_s, beta_e = 0.45, 0.35, 0.20
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
                ram_out=ram_out,
            )
            self.last_target_vx_axis = target
            return theta

        beta_v, beta_s, beta_e = 0.7, 0.2, 0.1
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
            ram_out=ram_out,
        )
        self.last_target_vx_axis = target
        return theta

    def beta_to_vx_axis(self, beta_v, beta_s, beta_e):
        energy_vx_axis = 0.5 * (
            self.nominal_vx_axis + self.conservative_vx_axis
        )

        vx = (
            beta_v * self.nominal_vx_axis +
            beta_s * self.conservative_vx_axis +
            beta_e * energy_vx_axis
        )

        return clamp(vx, 0.0, self.nominal_vx_axis)

    def _theta(self, mode, walk, target_vx_axis, emergency_stop,
               beta_v, beta_s, beta_e, rho, sigma, roll, pitch, ram_out):
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

            "v_cmd": ram_out.get("v_cmd", 0.0),
            "v_meas": ram_out.get("v_meas", 0.0),
            "rho_v_inst": ram_out.get("rho_v_inst", 0.0),
            "rho_v_mean": ram_out.get("rho_v_mean", 0.0),
            "sigma_v": ram_out.get("sigma_v", 0.0),

            # Backward-compatible field.
            "rho_v": ram_out.get("rho_v_mean", 0.0),
        }
