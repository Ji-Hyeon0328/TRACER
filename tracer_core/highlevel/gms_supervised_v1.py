from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Dict

import numpy as np


@dataclass
class GMSSupervisedV1Output:
    terrain: str
    mission: str
    risk_level: str
    selected_style: str
    confidence: float
    probabilities: Dict[str, float]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def _risk_level(ram_risk: float, recovery_needed: float) -> str:
    if ram_risk >= 0.50 or recovery_needed >= 0.50:
        return "high"
    if ram_risk >= 0.20 or recovery_needed >= 0.20:
        return "mid"
    return "low"


class GMSSupervisedV1:
    def __init__(self, model_path: str | Path = "configs/learned_models/tracer_gms_supervised_v1_model.json"):
        self.model_path = Path(model_path)
        self.model = json.loads(self.model_path.read_text())

        self.styles = list(self.model["styles"])
        self.missions = list(self.model["missions"])
        self.risk_levels = list(self.model["risk_levels"])
        self.terrains = list(self.model["terrains"])
        self.feature_names = list(self.model["feature_names"])

        self.mean = np.asarray(self.model["feature_mean"], dtype=np.float64)
        self.std = np.asarray(self.model["feature_std"], dtype=np.float64)
        self.std[self.std < 1e-8] = 1.0

        self.W = np.asarray(self.model["weights"], dtype=np.float64)
        self.b = np.asarray(self.model["bias"], dtype=np.float64)

    def _features(
        self,
        *,
        terrain: str,
        mission: str,
        risk_level: str,
        ram_risk: float,
        recovery_needed: float,
        beta_motion: float,
        beta_stability: float,
        beta_energy: float,
    ) -> np.ndarray:
        vals = {}

        for t in self.terrains:
            vals[f"terrain={t}"] = 1.0 if terrain == t else 0.0
        for m in self.missions:
            vals[f"mission={m}"] = 1.0 if mission == m else 0.0
        for r in self.risk_levels:
            vals[f"risk={r}"] = 1.0 if risk_level == r else 0.0

        vals["ram_risk"] = float(ram_risk)
        vals["recovery_needed"] = float(recovery_needed)
        vals["beta_motion"] = float(beta_motion)
        vals["beta_stability"] = float(beta_stability)
        vals["beta_energy"] = float(beta_energy)
        vals["ram_risk:beta_stability"] = float(ram_risk) * float(beta_stability)
        vals["ram_risk:beta_motion"] = float(ram_risk) * float(beta_motion)
        vals["recovery_needed:beta_stability"] = float(recovery_needed) * float(beta_stability)

        return np.asarray([vals.get(k, 0.0) for k in self.feature_names], dtype=np.float64)

    def select(
        self,
        *,
        terrain: str,
        mission: str = "deploy",
        ram_risk: float = 0.0,
        recovery_needed: float = 0.0,
        beta_motion: float = 0.34,
        beta_stability: float = 0.56,
        beta_energy: float = 0.10,
    ) -> GMSSupervisedV1Output:
        mission = str(mission or "deploy").strip().lower()
        if mission not in self.missions:
            mission = "deploy"

        risk = _risk_level(float(ram_risk), float(recovery_needed))

        x = self._features(
            terrain=str(terrain),
            mission=mission,
            risk_level=risk,
            ram_risk=float(ram_risk),
            recovery_needed=float(recovery_needed),
            beta_motion=float(beta_motion),
            beta_stability=float(beta_stability),
            beta_energy=float(beta_energy),
        )
        xs = (x - self.mean) / self.std
        p = _softmax(xs @ self.W + self.b)

        idx = int(np.argmax(p))
        probs = {style: float(p[i]) for i, style in enumerate(self.styles)}

        return GMSSupervisedV1Output(
            terrain=str(terrain),
            mission=mission,
            risk_level=risk,
            selected_style=self.styles[idx],
            confidence=float(p[idx]),
            probabilities=probs,
        )
