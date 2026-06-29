from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


DEFAULT_RAM_SCALAR_V3_MODEL = "configs/learned_models/tracer_ram_scalar_v3_model.pt"


@dataclass
class RAMScalarV3Output:
    probabilities: Dict[str, float]
    risk: float
    good_prob: float
    bad_prob: float
    feature_names: List[str]
    label_names: List[str]


class _RAMGRUV3:
    """Small wrapper that creates the torch model lazily.

    We keep torch imports inside the class so that ROS2/system-python users can
    import this module without immediately requiring torch. Actual inference
    still requires a Python environment with torch installed.
    """

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        import torch
        import torch.nn as nn

        class RAMGRUV2(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim),
                )

            def forward(self, x):
                _, h = self.gru(x)
                z = h[-1]
                return self.head(z)

        self.torch = torch
        self.model = RAMGRUV2(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_logits(self, x_np: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = torch.tensor(x_np, dtype=torch.float32)
            logits = self.model(x)
            return logits.cpu().numpy()


class RAMScalarV3:
    def __init__(self, model_path: str | Path = DEFAULT_RAM_SCALAR_V3_MODEL):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"RAM scalar v3 model not found: {self.model_path}")

        import torch

        try:
            artifact = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(self.model_path, map_location="cpu")

        self.input_dim = int(artifact["input_dim"])
        self.window = int(artifact["window"])
        self.hidden_dim = int(artifact["hidden_dim"])
        self.label_names = [str(x) for x in artifact["label_names"]]
        self.feature_mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(artifact["feature_std"], dtype=np.float32)
        self.feature_std[self.feature_std < 1e-6] = 1.0
        self.active_head_mask = [float(x) for x in artifact.get("active_head_mask", [1.0] * len(self.label_names))]

        # The model artifact currently does not store feature names directly.
        # Keep the canonical scalar v3 order here.
        self.feature_names = [
            "mpc_vx",
            "mpc_yaw",
            "mpc_body_height",
            "mpc_clearance",
            "mpc_enable",
            "beta_motion",
            "beta_stability",
            "beta_energy",
            "proprio_abs_mean",
            "proprio_abs_max",
            "proprio_base_x",
            "proprio_base_y",
            "proprio_base_z",
            "odom_x",
            "odom_y",
            "odom_z",
            "odom_vx",
            "age_mpc",
            "age_beta",
            "age_proprio",
            "age_odom",
            "age_debug",
        ]

        if len(self.feature_names) != self.input_dim:
            raise RuntimeError(
                f"feature_names length mismatch: {len(self.feature_names)} vs input_dim={self.input_dim}"
            )

        self._net = _RAMGRUV3(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            out_dim=len(self.label_names),
        )
        self._net.load_state_dict(artifact["model_state_dict"])

    def _sanitize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = np.clip(x, -1000.0, 1000.0)
        return x

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = self._sanitize(x)
        xs = (x - self.feature_mean.reshape(1, -1)) / self.feature_std.reshape(1, -1)
        xs = np.nan_to_num(xs, nan=0.0, posinf=10.0, neginf=-10.0)
        xs = np.clip(xs, -20.0, 20.0)
        return xs.astype(np.float32)

    def feature_vector_from_mapping(self, row: Mapping[str, float]) -> np.ndarray:
        vals = []
        for name in self.feature_names:
            try:
                vals.append(float(row.get(name, 0.0)))
            except Exception:
                vals.append(0.0)
        return np.asarray(vals, dtype=np.float32)

    def prepare_window(self, rows: Sequence[Mapping[str, float]] | np.ndarray) -> np.ndarray:
        if isinstance(rows, np.ndarray):
            x = np.asarray(rows, dtype=np.float32)
        else:
            x = np.asarray([self.feature_vector_from_mapping(r) for r in rows], dtype=np.float32)

        if x.ndim != 2:
            raise ValueError(f"Expected [T,D] window, got shape={x.shape}")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got shape={x.shape}")

        # Use the latest self.window rows; left-pad with the earliest row if short.
        if x.shape[0] >= self.window:
            x = x[-self.window :]
        else:
            pad_n = self.window - x.shape[0]
            if x.shape[0] == 0:
                pad = np.zeros((pad_n, self.input_dim), dtype=np.float32)
            else:
                pad = np.repeat(x[:1], pad_n, axis=0)
            x = np.concatenate([pad, x], axis=0)

        return self._normalize(x)

    def predict_window(self, rows: Sequence[Mapping[str, float]] | np.ndarray) -> RAMScalarV3Output:
        x = self.prepare_window(rows)
        logits = self._net.predict_logits(x.reshape(1, self.window, self.input_dim))[0]
        probs_arr = 1.0 / (1.0 + np.exp(-logits))
        probs = {name: float(probs_arr[i]) for i, name in enumerate(self.label_names)}

        fallen = probs.get("future_fallen_height", 0.0)
        low_h = probs.get("future_low_height", 0.0)
        low_prog = probs.get("future_low_progress", 0.0)
        bad = probs.get("future_bad_locomotion", 0.0)
        good = probs.get("future_good_locomotion", 0.0)

        risk = float(max(fallen, low_h, low_prog, bad, 1.0 - good))

        return RAMScalarV3Output(
            probabilities=probs,
            risk=risk,
            good_prob=float(good),
            bad_prob=float(bad),
            feature_names=list(self.feature_names),
            label_names=list(self.label_names),
        )
