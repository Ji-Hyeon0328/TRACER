from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math


@dataclass
class ObjectiveSelectorInput:
    terrain_key: str = "unknown"
    world: str = "unknown"
    terrain_family: str = "unknown"
    semantic_prior: str = "unknown"
    style_prior: str = "unknown"

    ramgate_enabled: bool = False
    ramgate_calibrated: bool = False
    ramgate_observed: bool = False
    ram_monitor_observed: bool = False

    ram_ctrl_risk: float = 0.0
    ram_fallen_prob: float = 0.0
    ram_recovery_prob: float = 0.0

    gate_stable_prob: float = 0.0
    gate_caution_prob: float = 0.0
    gate_unstable_prob: float = 0.0
    gate_override_prob: float = 0.0

    gate_first_action: str = ""
    gate_last_action: str = ""
    calibration_first: str = ""
    calibration_last: str = ""


@dataclass
class ObjectiveSelectorOutput:
    semantic_target: str
    deploy_label: str
    beta_v: float
    beta_s: float
    beta_e: float
    scores: dict[str, float]
    safety_override: bool
    reason: str

    @property
    def beta(self) -> tuple[float, float, float]:
        return (self.beta_v, self.beta_s, self.beta_e)


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _b(x: Any) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    return 1.0 if str(x).strip().lower() in {"1", "true", "yes", "y"} else 0.0


def infer_terrain_family(terrain_key: str, world: str = "", style: str = "") -> str:
    text = f"{terrain_key} {world} {style}".lower()

    if "slippery" in text or "slip" in text or "micro_brake" in text:
        return "slippery"
    if "sponge" in text or "soft" in text:
        return "sponge"
    if "downhill" in text or "downslope" in text:
        return "downslope"
    if "rough" in text:
        return "rough"
    if "slope" in text:
        return "slope"
    if "flat" in text or "earth" in text:
        return "flat"
    if "backstep" in text:
        return "backstep"
    return "unknown"


class RuntimeBaselineObjectiveSelector:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model = json.loads(self.model_path.read_text())

    @classmethod
    def from_default_model(cls, root: str | Path = ".") -> "RuntimeBaselineObjectiveSelector":
        root = Path(root)
        return cls(
            root
            / "data/preference_datasets/tracer_objective_selector_runtime_v0_baseline_model.json"
        )

    def encode(self, x: ObjectiveSelectorInput) -> list[float]:
        model = self.model
        vocab = model["vocab"]

        row = {
            "terrain_family": x.terrain_family or infer_terrain_family(x.terrain_key, x.world, x.style_prior),
            "semantic_prior": x.semantic_prior,
            "style_prior": x.style_prior,
            "gate_first_action": x.gate_first_action,
            "gate_last_action": x.gate_last_action,
            "calibration_first": x.calibration_first,
            "calibration_last": x.calibration_last,
            "ramgate_enabled": x.ramgate_enabled,
            "ramgate_calibrated": x.ramgate_calibrated,
            "ramgate_observed": x.ramgate_observed,
            "ram_monitor_observed": x.ram_monitor_observed,
            "ram_ctrl_risk": x.ram_ctrl_risk,
            "ram_fallen_prob": x.ram_fallen_prob,
            "ram_recovery_prob": x.ram_recovery_prob,
            "gate_stable_prob": x.gate_stable_prob,
            "gate_caution_prob": x.gate_caution_prob,
            "gate_unstable_prob": x.gate_unstable_prob,
            "gate_override_prob": x.gate_override_prob,
        }

        out: list[float] = []

        for key in model["categorical_features"]:
            val = str(row.get(key, ""))
            values = vocab[key]
            out.extend([1.0 if val == v else 0.0 for v in values])

        for key in model["bool_features"]:
            out.append(_b(row.get(key, False)))

        for key in model["numeric_features"]:
            out.append(_f(row.get(key, 0.0), 0.0))

        return out

    def _standardize(self, raw: list[float]) -> list[float]:
        means = self.model["feature_mean"]
        stds = self.model["feature_std"]
        return [(v - m) / s for v, m, s in zip(raw, means, stds)]

    @staticmethod
    def _dist2(a: list[float], b: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def safety_override_label(self, x: ObjectiveSelectorInput) -> str | None:
        model = self.model
        thresholds = model.get("safety_override_thresholds", {})

        ram_fallen_thr = _f(thresholds.get("ram_fallen_prob", 0.50), 0.50)
        ram_ctrl_thr = _f(thresholds.get("ram_ctrl_risk", 0.75), 0.75)
        ram_recovery_thr = _f(thresholds.get("ram_recovery_prob", 0.60), 0.60)
        gate_unstable_thr = _f(thresholds.get("gate_unstable_prob", 0.50), 0.50)
        gate_override_thr = _f(thresholds.get("gate_override_prob", 0.50), 0.50)

        safety_actions = set(model.get("safety_actions", []))

        strong_risk = (
            x.ram_fallen_prob >= ram_fallen_thr
            or x.ram_ctrl_risk >= ram_ctrl_thr
            or x.ram_recovery_prob >= ram_recovery_thr
            or x.gate_unstable_prob >= gate_unstable_thr
            or x.gate_override_prob >= gate_override_thr
        )

        action_risk = (
            x.gate_first_action in safety_actions
            or x.gate_last_action in safety_actions
        ) and (
            x.ram_fallen_prob >= 0.30
            or x.ram_ctrl_risk >= 0.25
            or x.gate_unstable_prob >= 0.30
            or x.gate_override_prob >= 0.30
        )

        if strong_risk or action_risk:
            return model.get("safety_override_label", "no_valid_forward_recovery_needed")

        return None

    def predict(self, x: ObjectiveSelectorInput) -> ObjectiveSelectorOutput:
        raw = self.encode(x)
        z = self._standardize(raw)

        scores = {}
        for label, centroid in self.model["centroids"].items():
            scores[label] = -self._dist2(z, centroid)

        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        forced = self.safety_override_label(x)
        safety_override = forced is not None

        if forced is not None:
            pred = forced
            scores["__safety_override__"] = 999.0

        deploy = self.model["deploy_by_label"][pred]
        beta = self.model["beta_by_label"][pred]
        s = sum(beta)
        if s > 1e-8:
            beta = [v / s for v in beta]

        reason = "safety_override" if safety_override else "nearest_centroid"

        return ObjectiveSelectorOutput(
            semantic_target=pred,
            deploy_label=deploy,
            beta_v=float(beta[0]),
            beta_s=float(beta[1]),
            beta_e=float(beta[2]),
            scores={k: float(v) for k, v in scores.items()},
            safety_override=safety_override,
            reason=reason,
        )


def objective_selector_input_from_dict(d: dict[str, Any]) -> ObjectiveSelectorInput:
    terrain_key = str(d.get("terrain_key", "unknown"))
    world = str(d.get("world", "unknown"))
    style_prior = str(d.get("style_prior", d.get("style", "unknown")))
    terrain_family = str(
        d.get("terrain_family")
        or infer_terrain_family(terrain_key, world, style_prior)
    )

    return ObjectiveSelectorInput(
        terrain_key=terrain_key,
        world=world,
        terrain_family=terrain_family,
        semantic_prior=str(d.get("semantic_prior", d.get("semantic", "unknown"))),
        style_prior=style_prior,
        ramgate_enabled=bool(d.get("ramgate_enabled", False)),
        ramgate_calibrated=bool(d.get("ramgate_calibrated", False)),
        ramgate_observed=bool(d.get("ramgate_observed", False)),
        ram_monitor_observed=bool(d.get("ram_monitor_observed", False)),
        ram_ctrl_risk=_f(d.get("ram_ctrl_risk", d.get("control_risk", 0.0))),
        ram_fallen_prob=_f(d.get("ram_fallen_prob", d.get("fallen_prob", 0.0))),
        ram_recovery_prob=_f(d.get("ram_recovery_prob", d.get("recovery_prob", 0.0))),
        gate_stable_prob=_f(d.get("gate_stable_prob", 0.0)),
        gate_caution_prob=_f(d.get("gate_caution_prob", 0.0)),
        gate_unstable_prob=_f(d.get("gate_unstable_prob", 0.0)),
        gate_override_prob=_f(d.get("gate_override_prob", d.get("would_override", 0.0))),
        gate_first_action=str(d.get("gate_first_action", d.get("gate_action", ""))),
        gate_last_action=str(d.get("gate_last_action", d.get("gate_action", ""))),
        calibration_first=str(d.get("calibration_first", "")),
        calibration_last=str(d.get("calibration_last", "")),
    )
