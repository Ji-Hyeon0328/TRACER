from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import json
import math

import numpy as np


STYLES = ["fast", "cautious", "high_clearance"]
OBJECTIVES = ["motion_objective", "stability_objective", "deploy_objective"]
TERRAINS = [
    "flat_normal",
    "rough_mid",
    "slope_5deg",
    "slippery_downslope_5deg_forward",
    "sponge_firm_downslope_5deg_forward",
]


@dataclass(frozen=True)
class BetaCandidate:
    name: str
    beta_motion: float
    beta_stability: float
    beta_energy: float
    objective_name: str


@dataclass
class ObjectiveSelectorV1Output:
    terrain: str
    mission_mode: str
    selected_beta_name: str
    beta_motion: float
    beta_stability: float
    beta_energy: float
    predicted_style: str
    score: float
    confidence: float
    reason: str


def _f(x, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def _sigmoid(x: float) -> float:
    x = max(-40.0, min(40.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _one_hot(name: str, choices: List[str]) -> List[float]:
    return [1.0 if name == c else 0.0 for c in choices]


def _candidate_metrics(row: dict) -> dict:
    success = _f(row.get("success_rate"))
    distance = _f(row.get("distance_mean"))
    progress = _f(row.get("progress_mean"))
    deploy = _f(row.get("deploy_mean"))
    fallen = _f(row.get("fallen_p90_max"))
    fresh = _f(row.get("fresh1_min"))

    stability_score = 0.45 * success + 0.25 * fresh + 0.30 * max(0.0, 1.0 - fallen)
    motion_score = 0.65 * distance + 0.35 * progress
    deploy_score = deploy
    energy_score = max(0.0, 1.0 - min(1.0, distance))

    return {
        "success_rate": success,
        "distance_mean": distance,
        "progress_mean": progress,
        "deploy_mean": deploy,
        "fallen_p90_max": fallen,
        "fresh1_min": fresh,
        "motion_score_proxy": motion_score,
        "stability_score_proxy": stability_score,
        "deploy_score_proxy": deploy_score,
        "energy_score_proxy": energy_score,
    }


def _build_features(
    *,
    feature_names: List[str],
    terrain: str,
    objective_name: str,
    beta_motion: float,
    beta_stability: float,
    beta_energy: float,
    style: str,
    aggregate_row: dict,
) -> np.ndarray:
    metrics = _candidate_metrics(aggregate_row)

    vals = {
        f"terrain={terrain}": 1.0,
        f"objective={objective_name}": 1.0,
        "beta_motion": beta_motion,
        "beta_stability": beta_stability,
        "beta_energy": beta_energy,
        f"style={style}": 1.0,
        f"style={style}:beta_motion": beta_motion,
        f"style={style}:beta_stability": beta_stability,
        f"style={style}:beta_energy": beta_energy,
    }

    for k, v in metrics.items():
        vals[k] = v
        vals[f"beta_motion:{k}"] = beta_motion * v
        vals[f"beta_stability:{k}"] = beta_stability * v
        vals[f"beta_energy:{k}"] = beta_energy * v

    return np.asarray([vals.get(name, 0.0) for name in feature_names], dtype=np.float64)


class ObjectiveSelectorV1:
    def __init__(
        self,
        *,
        model_path: str | Path = "configs/learned_models/tracer_preference_objective_irl_v1_model.json",
        aggregate_csv: str | Path = "reports/tracer_objective_selector_data_v1_core3_hard3_aggregate_by_style_clean.csv",
    ):
        self.model_path = Path(model_path)
        self.aggregate_csv = Path(aggregate_csv)

        self.model = json.loads(self.model_path.read_text())
        self.feature_names = list(self.model["feature_names"])
        self.weights = np.asarray(self.model["weights"], dtype=np.float64)
        self.bias = float(self.model["bias"])
        self.mean = np.asarray(self.model["feature_mean"], dtype=np.float64)
        self.std = np.asarray(self.model["feature_std"], dtype=np.float64)
        self.std[self.std < 1e-8] = 1.0

        self.aggregate: Dict[Tuple[str, str], dict] = {}
        with self.aggregate_csv.open(newline="") as fp:
            for row in csv.DictReader(fp):
                self.aggregate[(row["terrain"], row["style"])] = row

        self.beta_candidates = [
            BetaCandidate("motion", 0.70, 0.20, 0.10, "motion_objective"),
            BetaCandidate("balanced", 0.45, 0.40, 0.15, "deploy_objective"),
            BetaCandidate("deploy", 0.34, 0.56, 0.10, "deploy_objective"),
            BetaCandidate("stability", 0.15, 0.75, 0.10, "stability_objective"),
            BetaCandidate("safe", 0.08, 0.82, 0.10, "stability_objective"),
        ]

    def score_style(self, terrain: str, beta: BetaCandidate, style: str) -> Optional[float]:
        row = self.aggregate.get((terrain, style))
        if row is None:
            return None

        x = _build_features(
            feature_names=self.feature_names,
            terrain=terrain,
            objective_name=beta.objective_name,
            beta_motion=beta.beta_motion,
            beta_stability=beta.beta_stability,
            beta_energy=beta.beta_energy,
            style=style,
            aggregate_row=row,
        )
        xs = (x - self.mean) / self.std
        return float(xs @ self.weights + self.bias)

    def _beta_by_name(self, name: str) -> BetaCandidate:
        for beta in self.beta_candidates:
            if beta.name == name:
                return beta
        raise KeyError(name)

    def _select_beta_from_mission_and_risk(
        self,
        *,
        mission_mode: str,
        ram_risk: float,
        recovery_needed: float,
    ) -> tuple[BetaCandidate, str]:
        """Choose β from mission/risk priors.

        Important: the preference-IRL reward model scores styles for a given β.
        Its raw scores are not calibrated for directly comparing different β
        candidates. Therefore β is anchored by mission/risk first, and the
        learned reward model is used to predict/evaluate the style under that β.
        """
        high_risk = ram_risk >= 0.50 or recovery_needed >= 0.50
        mid_risk = ram_risk >= 0.20 or recovery_needed >= 0.20

        if high_risk:
            return self._beta_by_name("safe"), "risk_high_select_safe_beta"
        if mid_risk:
            return self._beta_by_name("stability"), "risk_mid_select_stability_beta"

        if mission_mode in {"motion", "fast", "speed"}:
            return self._beta_by_name("motion"), "mission_motion_select_motion_beta"
        if mission_mode in {"stability", "safe", "safety"}:
            return self._beta_by_name("stability"), "mission_stability_select_stability_beta"

        return self._beta_by_name("deploy"), "mission_deploy_select_deploy_beta"

    def select(
        self,
        *,
        terrain: str,
        mission_mode: str = "deploy",
        ram_risk: float = 0.0,
        recovery_needed: float = 0.0,
    ) -> ObjectiveSelectorV1Output:
        mission_mode = str(mission_mode or "deploy").strip().lower()
        ram_risk = float(ram_risk)
        recovery_needed = float(recovery_needed)

        selected_beta, beta_reason = self._select_beta_from_mission_and_risk(
            mission_mode=mission_mode,
            ram_risk=ram_risk,
            recovery_needed=recovery_needed,
        )

        # High risk should not keep requesting fast style even when the style
        # reward model still likes fast on easy terrain.
        high_risk = ram_risk >= 0.50 or recovery_needed >= 0.50
        mid_risk = ram_risk >= 0.20 or recovery_needed >= 0.20
        if high_risk:
            allowed_styles = ["cautious", "high_clearance"]
            style_reason = "risk_high_block_fast_style"
        elif mid_risk:
            allowed_styles = ["cautious", "high_clearance"]
            style_reason = "risk_mid_block_fast_style"
        else:
            allowed_styles = list(STYLES)
            style_reason = "risk_low_allow_all_styles"

        style_scores = []
        for style in allowed_styles:
            s = self.score_style(terrain, selected_beta, style)
            if s is not None:
                style_scores.append((style, s))

        if not style_scores:
            beta = BetaCandidate("safe_fallback", 0.08, 0.82, 0.10, "stability_objective")
            return ObjectiveSelectorV1Output(
                terrain=terrain,
                mission_mode=mission_mode,
                selected_beta_name=beta.name,
                beta_motion=beta.beta_motion,
                beta_stability=beta.beta_stability,
                beta_energy=beta.beta_energy,
                predicted_style="cautious",
                score=0.0,
                confidence=0.0,
                reason="fallback_no_aggregate_for_terrain",
            )

        style_scores.sort(key=lambda kv: kv[1], reverse=True)
        best_style, best_score = style_scores[0]
        second_score = style_scores[1][1] if len(style_scores) > 1 else best_score - 1.0
        confidence = _sigmoid(best_score - second_score)

        reason = (
            f"mission={mission_mode}, ram_risk={ram_risk:.3f}, "
            f"recovery_needed={recovery_needed:.3f}, "
            f"{beta_reason}, {style_reason}, "
            f"style_candidate_count={len(style_scores)}"
        )

        return ObjectiveSelectorV1Output(
            terrain=terrain,
            mission_mode=mission_mode,
            selected_beta_name=selected_beta.name,
            beta_motion=selected_beta.beta_motion,
            beta_stability=selected_beta.beta_stability,
            beta_energy=selected_beta.beta_energy,
            predicted_style=best_style,
            score=best_score,
            confidence=confidence,
            reason=reason,
        )
