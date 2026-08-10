from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.97
    gae_lambda: float = 0.95

    clip_ratio: float = 0.20

    learning_rate: float = 3e-4

    value_coef: float = 0.50
    entropy_coef: float = 0.01

    update_epochs: int = 4
    minibatch_size: int = 16

    max_grad_norm: float = 0.50


def _atanh(
    x: torch.Tensor,
) -> torch.Tensor:
    x = torch.clamp(
        x,
        -1.0 + 1e-6,
        1.0 - 1e-6,
    )

    return 0.5 * (
        torch.log1p(x)
        - torch.log1p(-x)
    )


class PPOActorCritic(nn.Module):
    """
    Continuous M7-v0 actor-critic.

    Actor:
      Gaussian in unconstrained z-space
      followed by tanh(z) -> [-1,1]^act_dim.

    Critic:
      scalar V(s).

    Initial std is intentionally small so M7 exploration
    starts around the characterized nominal command.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int = 4,
        hidden_sizes: Sequence[int] = (
            128,
            128,
        ),
        initial_std: float = 0.15,
    ) -> None:
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_sizes = tuple(
            int(x)
            for x in hidden_sizes
        )

        if self.obs_dim <= 0:
            raise ValueError(
                "obs_dim must be > 0"
            )

        if self.act_dim <= 0:
            raise ValueError(
                "act_dim must be > 0"
            )

        if initial_std <= 0.0:
            raise ValueError(
                "initial_std must be > 0"
            )

        layers = []

        in_dim = self.obs_dim

        for hidden in self.hidden_sizes:
            layers.extend(
                [
                    nn.Linear(
                        in_dim,
                        hidden,
                    ),
                    nn.Tanh(),
                ]
            )

            in_dim = hidden

        self.body = nn.Sequential(
            *layers
        )

        self.mu_head = nn.Linear(
            in_dim,
            self.act_dim,
        )

        self.value_head = nn.Linear(
            in_dim,
            1,
        )

        self.log_std = nn.Parameter(
            torch.full(
                (self.act_dim,),
                float(
                    math.log(
                        initial_std
                    )
                ),
                dtype=torch.float32,
            )
        )

        # Zero initial actor mean:
        #
        #   normalized action 0
        #       ->
        #   characterized M7 nominal command.
        nn.init.zeros_(
            self.mu_head.weight
        )

        nn.init.zeros_(
            self.mu_head.bias
        )

    def _features(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        return self.body(obs)

    def distribution_and_value(
        self,
        obs: torch.Tensor,
    ):
        h = self._features(obs)

        mu = self.mu_head(h)

        log_std = torch.clamp(
            self.log_std,
            -5.0,
            1.0,
        )

        std = torch.exp(
            log_std
        ).expand_as(mu)

        dist = torch.distributions.Normal(
            mu,
            std,
        )

        value = self.value_head(
            h
        ).squeeze(-1)

        return (
            dist,
            value,
        )

    @staticmethod
    def _squashed_log_prob(
        dist,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        base_log_prob = (
            dist.log_prob(z)
            .sum(dim=-1)
        )

        correction = torch.log(
            1.0
            - action * action
            + 1e-6
        ).sum(dim=-1)

        return (
            base_log_prob
            - correction
        )

    @torch.no_grad()
    def act(
        self,
        obs,
        *,
        deterministic: bool = False,
    ):
        obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
        )

        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        dist, value = (
            self.distribution_and_value(
                obs_t
            )
        )

        if deterministic:
            z = dist.mean
        else:
            z = dist.sample()

        action = torch.tanh(z)

        log_prob = (
            self._squashed_log_prob(
                dist,
                z,
                action,
            )
        )

        return (
            action[0]
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            ),

            float(
                log_prob[0]
                .cpu()
                .item()
            ),

            float(
                value[0]
                .cpu()
                .item()
            ),
        )

    @torch.no_grad()
    def value(
        self,
        obs,
    ) -> float:
        obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
        )

        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        _, value = (
            self.distribution_and_value(
                obs_t
            )
        )

        return float(
            value[0]
            .cpu()
            .item()
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ):
        dist, value = (
            self.distribution_and_value(
                obs
            )
        )

        clipped_action = torch.clamp(
            action,
            -1.0 + 1e-6,
            1.0 - 1e-6,
        )

        z = _atanh(
            clipped_action
        )

        log_prob = (
            self._squashed_log_prob(
                dist,
                z,
                clipped_action,
            )
        )

        # Exact entropy after tanh has no simple closed form.
        # For the small M7 entropy bonus we use the underlying
        # Gaussian entropy as the exploration proxy.
        entropy_proxy = (
            dist.entropy()
            .sum(dim=-1)
        )

        return (
            log_prob,
            entropy_proxy,
            value,
        )


class PPORolloutBuffer:
    """
    Small on-policy buffer with explicit next-value bootstrap.

    `continuation` controls GAE recursion across the NEXT
    transition. It must be zero at episode boundaries and at
    rollout-buffer boundaries.

    `next_value` may still bootstrap a time-limit truncation.
    """

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.next_values = []
        self.continuations = []

    def __len__(self):
        return len(
            self.rewards
        )

    def add(
        self,
        *,
        obs,
        action,
        reward: float,
        value: float,
        log_prob: float,
        next_value: float,
        continuation: float,
    ):
        self.obs.append(
            np.asarray(
                obs,
                dtype=np.float32,
            ).copy()
        )

        self.actions.append(
            np.asarray(
                action,
                dtype=np.float32,
            ).copy()
        )

        self.rewards.append(
            float(reward)
        )

        self.values.append(
            float(value)
        )

        self.log_probs.append(
            float(log_prob)
        )

        self.next_values.append(
            float(next_value)
        )

        self.continuations.append(
            float(continuation)
        )

    def close_rollout_boundary(
        self,
    ):
        if self.continuations:
            self.continuations[-1] = 0.0

    def as_batch(
        self,
        *,
        cfg: PPOConfig,
    ) -> dict[str, np.ndarray]:
        if len(self) == 0:
            raise ValueError(
                "empty PPO rollout buffer"
            )

        rewards = np.asarray(
            self.rewards,
            dtype=np.float32,
        )

        values = np.asarray(
            self.values,
            dtype=np.float32,
        )

        next_values = np.asarray(
            self.next_values,
            dtype=np.float32,
        )

        continuations = np.asarray(
            self.continuations,
            dtype=np.float32,
        )

        deltas = (
            rewards
            + float(cfg.gamma)
            * next_values
            - values
        )

        advantages = np.zeros_like(
            rewards,
            dtype=np.float32,
        )

        gae = 0.0

        for i in reversed(
            range(len(rewards))
        ):
            gae = (
                float(deltas[i])
                + float(cfg.gamma)
                * float(
                    cfg.gae_lambda
                )
                * float(
                    continuations[i]
                )
                * gae
            )

            advantages[i] = gae

        returns = (
            advantages
            + values
        )

        return {
            "obs":
                np.stack(
                    self.obs
                ).astype(
                    np.float32
                ),

            "actions":
                np.stack(
                    self.actions
                ).astype(
                    np.float32
                ),

            "old_log_probs":
                np.asarray(
                    self.log_probs,
                    dtype=np.float32,
                ),

            "advantages":
                advantages,

            "returns":
                returns,
        }


def ppo_update(
    *,
    model: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    cfg: PPOConfig,
) -> dict[str, float]:
    obs = torch.as_tensor(
        batch["obs"],
        dtype=torch.float32,
    )

    actions = torch.as_tensor(
        batch["actions"],
        dtype=torch.float32,
    )

    old_log_probs = torch.as_tensor(
        batch["old_log_probs"],
        dtype=torch.float32,
    )

    advantages = torch.as_tensor(
        batch["advantages"],
        dtype=torch.float32,
    )

    returns = torch.as_tensor(
        batch["returns"],
        dtype=torch.float32,
    )

    advantages = (
        advantages
        - advantages.mean()
    ) / (
        advantages.std(
            unbiased=False
        )
        + 1e-8
    )

    n = int(
        obs.shape[0]
    )

    mb = min(
        int(cfg.minibatch_size),
        n,
    )

    metrics = {
        "policy_loss": [],
        "value_loss": [],
        "entropy_proxy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }

    for _ in range(
        int(cfg.update_epochs)
    ):
        permutation = torch.randperm(
            n
        )

        for start in range(
            0,
            n,
            mb,
        ):
            idx = permutation[
                start:
                start + mb
            ]

            (
                new_log_prob,
                entropy_proxy,
                value,
            ) = model.evaluate_actions(
                obs[idx],
                actions[idx],
            )

            log_ratio = (
                new_log_prob
                - old_log_probs[idx]
            )

            ratio = torch.exp(
                log_ratio
            )

            adv = advantages[idx]

            unclipped = (
                ratio
                * adv
            )

            clipped = (
                torch.clamp(
                    ratio,
                    1.0
                    - float(
                        cfg.clip_ratio
                    ),
                    1.0
                    + float(
                        cfg.clip_ratio
                    ),
                )
                * adv
            )

            policy_loss = -torch.min(
                unclipped,
                clipped,
            ).mean()

            value_loss = F.mse_loss(
                value,
                returns[idx],
            )

            entropy_mean = (
                entropy_proxy.mean()
            )

            loss = (
                policy_loss
                + float(
                    cfg.value_coef
                )
                * value_loss
                - float(
                    cfg.entropy_coef
                )
                * entropy_mean
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(
                    cfg.max_grad_norm
                ),
            )

            optimizer.step()

            with torch.no_grad():
                approx_kl = (
                    old_log_probs[idx]
                    - new_log_prob
                ).mean()

                clip_fraction = (
                    (
                        torch.abs(
                            ratio - 1.0
                        )
                        > float(
                            cfg.clip_ratio
                        )
                    )
                    .float()
                    .mean()
                )

            metrics[
                "policy_loss"
            ].append(
                float(
                    policy_loss.item()
                )
            )

            metrics[
                "value_loss"
            ].append(
                float(
                    value_loss.item()
                )
            )

            metrics[
                "entropy_proxy"
            ].append(
                float(
                    entropy_mean.item()
                )
            )

            metrics[
                "approx_kl"
            ].append(
                float(
                    approx_kl.item()
                )
            )

            metrics[
                "clip_fraction"
            ].append(
                float(
                    clip_fraction.item()
                )
            )

    return {
        key:
            float(
                np.mean(values)
            )

        for key, values
        in metrics.items()
    }


def save_ppo_checkpoint(
    path,
    *,
    model: PPOActorCritic,
    cfg: PPOConfig,
    extra: dict | None = None,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema":
            "icra27_m7_online_ppo_v0",

        "obs_dim":
            model.obs_dim,

        "act_dim":
            model.act_dim,

        "hidden_sizes":
            list(
                model.hidden_sizes
            ),

        "ppo_config":
            asdict(cfg),

        "model_state_dict":
            model.state_dict(),

        "extra":
            dict(
                extra or {}
            ),
    }

    torch.save(
        payload,
        path,
    )


def load_ppo_checkpoint(
    path,
    *,
    map_location="cpu",
):
    path = Path(path)

    try:
        payload = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

    except TypeError:
        payload = torch.load(
            path,
            map_location=map_location,
        )

    if (
        payload.get("schema")
        != "icra27_m7_online_ppo_v0"
    ):
        raise ValueError(
            "unexpected PPO checkpoint schema"
        )

    cfg = PPOConfig(
        **payload[
            "ppo_config"
        ]
    )

    model = PPOActorCritic(
        obs_dim=int(
            payload["obs_dim"]
        ),

        act_dim=int(
            payload["act_dim"]
        ),

        hidden_sizes=tuple(
            int(x)
            for x in payload[
                "hidden_sizes"
            ]
        ),

        initial_std=0.15,
    )

    model.load_state_dict(
        payload[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        cfg,
        payload,
    )
