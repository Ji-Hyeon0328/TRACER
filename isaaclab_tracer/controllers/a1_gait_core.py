from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class A1GaitCoreCfg:
    counter_per_gait: float = 240.0
    counter_per_swing: float = 120.0

    # Same trot phase as A1CtrlStates::gait_counter_reset()
    # FL, FR, RL, RR
    init_gait_counter: tuple[float, float, float, float] = (0.0, 120.0, 120.0, 0.0)
    gait_counter_speed: tuple[float, float, float, float] = (2.0, 2.0, 2.0, 2.0)

    # Baseline default foot positions, body frame
    # columns: FL, FR, RL, RR
    default_foot_pos: tuple[tuple[float, float, float, float], ...] = (
        (0.17, 0.17, -0.17, -0.17),
        (0.15, -0.15, 0.15, -0.15),
        (-0.35, -0.35, -0.35, -0.35),
    )

    foot_delta_x_limit: float = 0.10
    foot_delta_y_limit: float = 0.10

    # From A1Params.h
    foot_swing_clearance1: float = 0.0
    foot_swing_clearance2: float = 0.4

    # IsaacLab env step is currently 0.02 s.
    control_dt: float = 0.02


class A1GaitCore:
    """Python port of the baseline A1 gait planner + Bezier swing trajectory.

    This module does NOT compute IK or torque yet.
    It only outputs desired foot targets and stance/swing masks.
    """

    def __init__(self, num_envs: int, device: torch.device | str, cfg: A1GaitCoreCfg | None = None):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg or A1GaitCoreCfg()

        self.default_foot_pos = torch.tensor(
            self.cfg.default_foot_pos,
            dtype=torch.float32,
            device=self.device,
        )  # (3, 4)

        self.gait_counter_speed = torch.tensor(
            self.cfg.gait_counter_speed,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 4)

        self.init_gait_counter = torch.tensor(
            self.cfg.init_gait_counter,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 4)

        self.gait_counter = self.init_gait_counter.repeat(self.num_envs, 1)
        self.foot_pos_start = self.default_foot_pos.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.foot_pos_target_last = self.default_foot_pos.unsqueeze(0).repeat(self.num_envs, 1, 1)

        # Env-specific default foot positions.
        # By default this uses the baseline constants, but for IsaacLab we should
        # overwrite it using FK(default_joint_pos) to avoid model mismatch.
        self.default_foot_pos_env = self.default_foot_pos.unsqueeze(0).repeat(self.num_envs, 1, 1)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.gait_counter[env_ids] = self.init_gait_counter
        self.foot_pos_start[env_ids] = self.default_foot_pos_env[env_ids]
        self.foot_pos_target_last[env_ids] = self.default_foot_pos_env[env_ids]

    def set_default_foot_pos(self, foot_pos_rel):
        """Override default foot positions.

        Args:
            foot_pos_rel:
                Either (3, 4) or (N, 3, 4), columns are FL, FR, RL, RR.
                For IsaacLab, pass FK(default_joint_pos).
        """
        if foot_pos_rel.dim() == 2:
            foot_pos_rel = foot_pos_rel.unsqueeze(0).repeat(self.num_envs, 1, 1)
        if foot_pos_rel.shape != (self.num_envs, 3, 4):
            raise ValueError(f"Expected {(self.num_envs, 3, 4)}, got {tuple(foot_pos_rel.shape)}")

        self.default_foot_pos_env = foot_pos_rel.to(self.device).detach().clone()
        self.foot_pos_start = self.default_foot_pos_env.clone()
        self.foot_pos_target_last = self.default_foot_pos_env.clone()

    def _bezier_curve(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # degree 4 Bezier with coefficients [1, 4, 6, 4, 1]
        # t: (N, 4)
        # p: (N, 4, 5)
        coeff = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=self.device).view(1, 1, 5)
        powers = torch.stack(
            [
                (1 - t) ** 4,
                t * (1 - t) ** 3,
                t**2 * (1 - t) ** 2,
                t**3 * (1 - t),
                t**4,
            ],
            dim=-1,
        )
        return torch.sum(coeff * powers * p, dim=-1)

    def _foot_pos_curve(self, t: torch.Tensor, foot_start: torch.Tensor, foot_final: torch.Tensor) -> torch.Tensor:
        # t: (N, 4)
        # foot_start, foot_final: (N, 3, 4)
        out = torch.zeros_like(foot_final)

        sx = foot_start[:, 0, :]
        sy = foot_start[:, 1, :]
        sz = foot_start[:, 2, :]

        fx = foot_final[:, 0, :]
        fy = foot_final[:, 1, :]
        fz = foot_final[:, 2, :]

        px = torch.stack([sx, sx, fx, fx, fx], dim=-1)
        py = torch.stack([sy, sy, fy, fy, fy], dim=-1)
        pz = torch.stack([sz, sz, fz, fz, fz], dim=-1)

        pz[:, :, 1] += self.cfg.foot_swing_clearance1
        pz[:, :, 2] += self.cfg.foot_swing_clearance2

        out[:, 0, :] = self._bezier_curve(t, px)
        out[:, 1, :] = self._bezier_curve(t, py)
        out[:, 2, :] = self._bezier_curve(t, pz)
        return out

    def step(
        self,
        foot_pos_cur_rel: torch.Tensor,
        root_lin_vel_body: torch.Tensor | None = None,
        root_lin_vel_cmd_body: torch.Tensor | None = None,
        movement_mode: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute gait phase and foot-space swing targets.

        Args:
            foot_pos_cur_rel: (N, 3, 4), current foot positions in body/yaw-aligned frame.
            root_lin_vel_body: (N, 3)
            root_lin_vel_cmd_body: (N, 3)
            movement_mode: if False, all legs are stance.

        Returns:
            dict with plan_contacts, swing_mask, spline_time, foot_pos_target_rel.
        """
        if root_lin_vel_body is None:
            root_lin_vel_body = torch.zeros(self.num_envs, 3, device=self.device)
        if root_lin_vel_cmd_body is None:
            root_lin_vel_cmd_body = torch.zeros(self.num_envs, 3, device=self.device)

        if not movement_mode:
            plan_contacts = torch.ones(self.num_envs, 4, dtype=torch.bool, device=self.device)
            self.gait_counter[:] = self.init_gait_counter
        else:
            self.gait_counter = torch.remainder(
                self.gait_counter + self.gait_counter_speed,
                self.cfg.counter_per_gait,
            )
            plan_contacts = self.gait_counter <= self.cfg.counter_per_swing

        # Baseline Raibert-style foot target.
        foot_final = self.default_foot_pos_env.clone()

        h = torch.mean(torch.abs(self.default_foot_pos_env[:, 2, :])).item()
        sqrt_h_g = (h / 9.8) ** 0.5

        swing_duration = (self.cfg.counter_per_swing / self.gait_counter_speed) * self.cfg.control_dt

        delta_x = sqrt_h_g * (root_lin_vel_body[:, 0:1] - root_lin_vel_cmd_body[:, 0:1])
        delta_x = delta_x + 0.5 * swing_duration * root_lin_vel_cmd_body[:, 0:1]
        delta_x = torch.clamp(delta_x, -self.cfg.foot_delta_x_limit, self.cfg.foot_delta_x_limit)

        delta_y = sqrt_h_g * (root_lin_vel_body[:, 1:2] - root_lin_vel_cmd_body[:, 1:2])
        delta_y = delta_y + 0.5 * swing_duration * root_lin_vel_cmd_body[:, 1:2]
        delta_y = torch.clamp(delta_y, -self.cfg.foot_delta_y_limit, self.cfg.foot_delta_y_limit)

        foot_final[:, 0, :] += delta_x
        foot_final[:, 1, :] += delta_y

        # In stance, baseline keeps refreshing foot_pos_start.
        stance_mask = plan_contacts
        self.foot_pos_start = torch.where(
            stance_mask.unsqueeze(1),
            foot_pos_cur_rel,
            self.foot_pos_start,
        )

        spline_time = torch.zeros(self.num_envs, 4, device=self.device)
        swing_mask = ~plan_contacts
        spline_time[swing_mask] = (
            (self.gait_counter[swing_mask] - self.cfg.counter_per_swing) / self.cfg.counter_per_swing
        )
        spline_time = torch.clamp(spline_time, 0.0, 1.0)

        foot_target = self._foot_pos_curve(spline_time, self.foot_pos_start, foot_final)
        self.foot_pos_target_last = foot_target

        return {
            "plan_contacts": plan_contacts,
            "swing_mask": swing_mask,
            "spline_time": spline_time,
            "foot_pos_target_rel": foot_target,
            "foot_pos_final_rel": foot_final,
            "gait_counter": self.gait_counter.clone(),
        }
