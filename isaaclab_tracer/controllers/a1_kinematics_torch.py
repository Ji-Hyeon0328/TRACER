import torch


class A1KinematicsTorch:
    """Torch FK + numerical IK for Unitree A1.

    Joint order assumptions:
    - IsaacLab order:
      [FL_hip, FR_hip, RL_hip, RR_hip,
       FL_thigh, FR_thigh, RL_thigh, RR_thigh,
       FL_calf, FR_calf, RL_calf, RR_calf]

    - Leg-major order:
      [FL_hip, FL_thigh, FL_calf,
       FR_hip, FR_thigh, FR_calf,
       RL_hip, RL_thigh, RL_calf,
       RR_hip, RR_thigh, RR_calf]
    """

    def __init__(self, device, upper_leg_length=0.22, lower_leg_length=0.21):
        self.device = torch.device(device)

        self.leg_offset_x = torch.tensor(
            [0.1805, 0.1805, -0.1805, -0.1805],
            dtype=torch.float32,
            device=self.device,
        )
        self.leg_offset_y = torch.tensor(
            [0.047, -0.047, 0.047, -0.047],
            dtype=torch.float32,
            device=self.device,
        )
        self.motor_offset = torch.tensor(
            [0.0838, -0.0838, 0.0838, -0.0838],
            dtype=torch.float32,
            device=self.device,
        )
        self.upper_leg_length = torch.full(
            (4,),
            float(upper_leg_length),
            dtype=torch.float32,
            device=self.device,
        )
        self.lower_leg_length = torch.full(
            (4,),
            float(lower_leg_length),
            dtype=torch.float32,
            device=self.device,
        )

        # IsaacLab -> leg-major
        self.isaac_to_leg_idx = torch.tensor(
            [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11],
            dtype=torch.long,
            device=self.device,
        )

        # leg-major -> IsaacLab
        self.leg_to_isaac_idx = torch.tensor(
            [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11],
            dtype=torch.long,
            device=self.device,
        )

        self.q_lower = torch.tensor(
            [-0.9, -1.4, -2.8],
            dtype=torch.float32,
            device=self.device,
        )
        self.q_upper = torch.tensor(
            [0.9, 2.2, -0.35],
            dtype=torch.float32,
            device=self.device,
        )

    def isaac_to_leg_order(self, q_isaac):
        return q_isaac[:, self.isaac_to_leg_idx]

    def leg_to_isaac_order(self, q_leg_major):
        return q_leg_major[:, self.leg_to_isaac_idx]

    def fk_leg_flat(self, q_leg, leg_ids):
        """FK for flattened legs.

        Args:
            q_leg: (B, 3), [hip, thigh, calf]
            leg_ids: (B,), 0 FL, 1 FR, 2 RL, 3 RR

        Returns:
            foot position in robot body frame, (B, 3)
        """
        q0 = q_leg[:, 0]
        q1 = q_leg[:, 1]
        q2 = q_leg[:, 2]

        lx = self.leg_offset_x[leg_ids]
        ly = self.leg_offset_y[leg_ids]
        mo = self.motor_offset[leg_ids]
        l1 = self.upper_leg_length[leg_ids]
        l2 = self.lower_leg_length[leg_ids]

        c0 = torch.cos(q0)
        s0 = torch.sin(q0)
        c1 = torch.cos(q1)
        s1 = torch.sin(q1)
        c12 = torch.cos(q1 + q2)
        s12 = torch.sin(q1 + q2)

        x = lx - l1 * s1 - l2 * s12
        y = ly + mo * c0 + s0 * (l1 * c1 + l2 * c12)
        z = mo * s0 - c0 * (l1 * c1 + l2 * c12)

        return torch.stack([x, y, z], dim=-1)

    def fk_leg_major(self, q_leg_major):
        """FK from leg-major 12-dim joint positions.

        Args:
            q_leg_major: (N, 12)

        Returns:
            foot positions, (N, 3, 4), columns FL, FR, RL, RR
        """
        n = q_leg_major.shape[0]
        q = q_leg_major.reshape(n, 4, 3).reshape(n * 4, 3)
        leg_ids = torch.arange(4, device=self.device).repeat(n)

        p = self.fk_leg_flat(q, leg_ids)
        return p.reshape(n, 4, 3).permute(0, 2, 1).contiguous()

    def fk_isaac(self, q_isaac):
        q_leg = self.isaac_to_leg_order(q_isaac)
        return self.fk_leg_major(q_leg)

    def ik_foot_targets(
        self,
        q_init_isaac,
        foot_target_rel,
        num_iters=18,
        damping=1.0e-3,
        step_size=1.0,
        eps=1.0e-4,
    ):
        """Damped least-squares numerical IK.

        Args:
            q_init_isaac: (N, 12), IsaacLab joint order.
            foot_target_rel: (N, 3, 4), desired foot positions in body frame.

        Returns:
            q_target_isaac: (N, 12), IsaacLab joint order.
        """
        n = q_init_isaac.shape[0]

        q_leg_major = self.isaac_to_leg_order(q_init_isaac).clone()
        q = q_leg_major.reshape(n * 4, 3).clone()

        target = foot_target_rel.permute(0, 2, 1).reshape(n * 4, 3)
        leg_ids = torch.arange(4, device=self.device).repeat(n)

        eye3 = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(n * 4, 1, 1)

        for _ in range(num_iters):
            p = self.fk_leg_flat(q, leg_ids)
            err = target - p

            cols = []
            for j in range(3):
                q_eps = q.clone()
                q_eps[:, j] += eps
                p_eps = self.fk_leg_flat(q_eps, leg_ids)
                cols.append((p_eps - p) / eps)

            # J: (B, 3, 3), where columns are partial derivatives wrt q_j
            J = torch.stack(cols, dim=-1)

            # dq = J^T (J J^T + lambda I)^-1 err
            A = torch.bmm(J, J.transpose(1, 2)) + damping * eye3
            y = torch.linalg.solve(A, err.unsqueeze(-1))
            dq = torch.bmm(J.transpose(1, 2), y).squeeze(-1)

            q = q + step_size * dq
            q = torch.max(torch.min(q, self.q_upper.view(1, 3)), self.q_lower.view(1, 3))

        q_leg_major = q.reshape(n, 12)
        return self.leg_to_isaac_order(q_leg_major)


def _a1_kin_jacobian_leg_flat(self, q_leg, leg_ids, eps=1.0e-4):
    """Numerical foot Jacobian for flattened legs.

    Args:
        q_leg: (B, 3)
        leg_ids: (B,)

    Returns:
        J: (B, 3, 3), foot position derivative wrt [hip, thigh, calf]
    """
    p0 = self.fk_leg_flat(q_leg, leg_ids)

    cols = []
    for j in range(3):
        q_eps = q_leg.clone()
        q_eps[:, j] += eps
        p_eps = self.fk_leg_flat(q_eps, leg_ids)
        cols.append((p_eps - p0) / eps)

    return torch.stack(cols, dim=-1)


def _a1_kin_foot_forces_to_joint_torques(self, q_isaac, foot_forces_body):
    """Map foot forces to joint torques.

    Args:
        q_isaac: (N, 12), IsaacLab joint order.
        foot_forces_body: (N, 3, 4), columns FL, FR, RL, RR.

    Returns:
        tau_isaac: (N, 12), IsaacLab joint order.
    """
    n = q_isaac.shape[0]

    q_leg_major = self.isaac_to_leg_order(q_isaac)
    q_flat = q_leg_major.reshape(n * 4, 3)

    leg_ids = torch.arange(4, device=self.device).repeat(n)
    J = self.jacobian_leg_flat(q_flat, leg_ids)

    f_flat = foot_forces_body.permute(0, 2, 1).reshape(n * 4, 3)
    tau_flat = torch.bmm(J.transpose(1, 2), f_flat.unsqueeze(-1)).squeeze(-1)

    tau_leg_major = tau_flat.reshape(n, 12)
    return self.leg_to_isaac_order(tau_leg_major)


A1KinematicsTorch.jacobian_leg_flat = _a1_kin_jacobian_leg_flat
A1KinematicsTorch.foot_forces_to_joint_torques = _a1_kin_foot_forces_to_joint_torques
