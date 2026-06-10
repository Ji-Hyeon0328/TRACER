from __future__ import annotations

import torch


class BatchedHistoryBuffer:
    def __init__(self, num_envs: int, history_len: int, feature_dim: int, device: str):
        self.num_envs = num_envs
        self.history_len = history_len
        self.feature_dim = feature_dim
        self.device = device
        self.data = torch.zeros(num_envs, history_len, feature_dim, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.data.zero_()
        else:
            self.data[env_ids] = 0.0

    def append(self, x: torch.Tensor):
        # x: [num_envs, feature_dim]
        self.data = torch.roll(self.data, shifts=-1, dims=1)
        self.data[:, -1, :] = x

    def get(self) -> torch.Tensor:
        return self.data
