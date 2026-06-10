from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TracerOutput:
    rho: torch.Tensor
    sigma: torch.Tensor
    beta: torch.Tensor
    theta: torch.Tensor
    vx_cmd: torch.Tensor
    yaw_cmd: torch.Tensor


class HistoryEncoderGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ELU(),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: [B, H, F]
        _, h = self.gru(history)
        h = h[-1]
        return self.proj(h)


class RAMHead(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 2),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = torch.sigmoid(self.net(z))
        rho = out[:, 0:1]
        sigma = out[:, 1:2]
        return rho, sigma


class ObjectiveSelector(nn.Module):
    def __init__(self, context_dim: int, goal_dim: int, latent_dim: int = 32):
        super().__init__()
        in_dim = context_dim + goal_dim + latent_dim + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        context: torch.Tensor,
        goal: torch.Tensor,
        latent: torch.Tensor,
        rho: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([context, goal, latent, rho, sigma], dim=-1)
        beta = torch.softmax(self.net(x), dim=-1)
        return beta


class ThetaDecoder(nn.Module):
    def __init__(self, context_dim: int, latent_dim: int = 32, theta_dim: int = 6):
        super().__init__()
        in_dim = context_dim + latent_dim + 2 + 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, theta_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        context: torch.Tensor,
        latent: torch.Tensor,
        rho: torch.Tensor,
        sigma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([context, latent, rho, sigma, beta], dim=-1)
        return self.net(x)


class ThetaToRefMapper(nn.Module):
    def __init__(self, max_vx: float = 0.12, max_yaw: float = 0.20):
        super().__init__()
        self.max_vx = max_vx
        self.max_yaw = max_yaw

    def forward(
        self,
        raw_goal_cmd: torch.Tensor,
        beta: torch.Tensor,
        theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # raw_goal_cmd: [B, 2] = [raw_vx, raw_yaw]
        raw_vx = raw_goal_cmd[:, 0:1]
        raw_yaw = raw_goal_cmd[:, 1:2]

        beta_v = beta[:, 0:1]
        beta_s = beta[:, 1:2]
        beta_e = beta[:, 2:3]

        # theta[:, 0] = vx_scale residual, theta[:, 1] = yaw_scale residual
        vx_theta_scale = 0.75 + 0.25 * theta[:, 0:1]
        yaw_theta_scale = 0.85 + 0.15 * theta[:, 1:2]

        conservative_vx = torch.clamp(raw_vx, -0.05, 0.05)
        energy_vx = 0.5 * (raw_vx + conservative_vx)

        vx_cmd = beta_v * raw_vx + beta_s * conservative_vx + beta_e * energy_vx
        vx_cmd = vx_cmd * vx_theta_scale
        vx_cmd = torch.clamp(vx_cmd, -self.max_vx, self.max_vx)

        # stability-heavy beta attenuates yaw
        yaw_scale = beta_v + 0.85 * beta_e + 0.65 * beta_s
        yaw_cmd = raw_yaw * yaw_scale * yaw_theta_scale
        yaw_cmd = torch.clamp(yaw_cmd, -self.max_yaw, self.max_yaw)

        return vx_cmd, yaw_cmd


class TracerHighLevel(nn.Module):
    def __init__(
        self,
        history_dim: int,
        context_dim: int,
        goal_dim: int = 2,
        latent_dim: int = 32,
        theta_dim: int = 6,
        max_vx: float = 0.12,
        max_yaw: float = 0.20,
    ):
        super().__init__()
        self.history_encoder = HistoryEncoderGRU(history_dim, 64, latent_dim)
        self.ram_head = RAMHead(latent_dim)
        self.objective_selector = ObjectiveSelector(context_dim, goal_dim, latent_dim)
        self.theta_decoder = ThetaDecoder(context_dim, latent_dim, theta_dim)
        self.ref_mapper = ThetaToRefMapper(max_vx=max_vx, max_yaw=max_yaw)

    def forward(
        self,
        history: torch.Tensor,
        context: torch.Tensor,
        raw_goal_cmd: torch.Tensor,
    ) -> TracerOutput:
        latent = self.history_encoder(history)
        rho, sigma = self.ram_head(latent)
        beta = self.objective_selector(context, raw_goal_cmd, latent, rho, sigma)
        theta = self.theta_decoder(context, latent, rho, sigma, beta)
        vx_cmd, yaw_cmd = self.ref_mapper(raw_goal_cmd, beta, theta)

        return TracerOutput(
            rho=rho,
            sigma=sigma,
            beta=beta,
            theta=theta,
            vx_cmd=vx_cmd,
            yaw_cmd=yaw_cmd,
        )
