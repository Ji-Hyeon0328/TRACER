from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import json


@dataclass
class TracerTaskSpec:
    name: str
    num_envs: int
    episode_length_sec: float
    control_dt: float
    decimation: int
    history_len: int
    history_feature_dim: int
    context_dim: int
    goal_dim: int
    action_dim: int
    max_vx: float
    max_yaw: float
    latent_dim: int
    theta_dim: int

    @classmethod
    def from_json(cls, path: str) -> "TracerTaskSpec":
        with open(path, "r") as f:
            cfg: Dict[str, Any] = json.load(f)

        obs = cfg["observation_dims"]
        action = cfg["action_space"]
        high = cfg["tracer_high_level"]

        return cls(
            name=cfg["name"],
            num_envs=int(cfg["num_envs"]),
            episode_length_sec=float(cfg["episode_length_sec"]),
            control_dt=float(cfg["control_dt"]),
            decimation=int(cfg["decimation"]),
            history_len=int(obs["history_len"]),
            history_feature_dim=int(obs["history_feature_dim"]),
            context_dim=int(obs["context_dim"]),
            goal_dim=int(obs["goal_dim"]),
            action_dim=int(action["dim"]),
            max_vx=float(high["max_vx"]),
            max_yaw=float(high["max_yaw"]),
            latent_dim=int(high["latent_dim"]),
            theta_dim=int(high["theta_dim"]),
        )
