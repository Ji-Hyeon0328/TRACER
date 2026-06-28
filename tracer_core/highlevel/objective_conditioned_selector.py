from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _f(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def _normalize_beta(beta: Any) -> dict[str, float]:
    if beta is None:
        return {"motion": 0.34, "stability": 0.56, "energy": 0.10}

    if isinstance(beta, dict):
        m = _f(beta.get("motion", beta.get("beta_motion", 0.0)))
        s = _f(beta.get("stability", beta.get("beta_stability", 0.0)))
        e = _f(beta.get("energy", beta.get("beta_energy", 0.0)))
    elif isinstance(beta, (list, tuple)) and len(beta) >= 3:
        m, s, e = _f(beta[0]), _f(beta[1]), _f(beta[2])
    else:
        m = _f(getattr(beta, "motion", 0.0))
        s = _f(getattr(beta, "stability", 0.0))
        e = _f(getattr(beta, "energy", 0.0))

    m = max(0.0, m)
    s = max(0.0, s)
    e = max(0.0, e)
    z = m + s + e
    if z <= 1e-9:
        return {"motion": 0.34, "stability": 0.56, "energy": 0.10}

    return {"motion": m / z, "stability": s / z, "energy": e / z}


def _beta_l2(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt(
        (a["motion"] - b["motion"]) ** 2
        + (a["stability"] - b["stability"]) ** 2
        + (a["energy"] - b["energy"]) ** 2
    )


class ObjectiveConditionedStyleSelector:
    """Runtime rule selector exported from objective-conditioned preference pairs.

    Given terrain_key and beta, this selector:
      1. normalizes beta,
      2. finds the nearest objective profile,
      3. returns the style/semantic rule for the terrain/objective.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model = json.loads(self.model_path.read_text())

        self.objective_profiles = self.model.get("objective_profiles", {})
        self.terrain_objective_rules = self.model.get("terrain_objective_rules", {})
        self.default_style_by_terrain = self.model.get("default_style_by_terrain", {})
        self.style_to_semantic = self.model.get("style_to_semantic", {})

        if not self.objective_profiles:
            raise RuntimeError(f"objective_profiles missing in {self.model_path}")
        if not self.terrain_objective_rules:
            raise RuntimeError(f"terrain_objective_rules missing in {self.model_path}")

    def nearest_objective_profile(self, beta: Any) -> tuple[str, float, dict[str, float]]:
        b = _normalize_beta(beta)
        best_name = None
        best_dist = 1e9

        for name, prof in self.objective_profiles.items():
            p = _normalize_beta(prof)
            d = _beta_l2(b, p)
            if d < best_dist:
                best_name = str(name)
                best_dist = d

        return best_name or "deploy_objective", float(best_dist), b

    def select(
        self,
        terrain_key: str,
        beta: Any,
        *,
        min_confidence: float = 0.0,
    ) -> dict[str, Any]:
        terrain = str(terrain_key)
        objective_name, objective_distance, beta_norm = self.nearest_objective_profile(beta)

        terrain_rules = self.terrain_objective_rules.get(terrain, {})
        rule = terrain_rules.get(objective_name)

        if rule is None:
            fallback_style = self.default_style_by_terrain.get(terrain, "cautious")
            semantic = self.style_to_semantic.get(
                fallback_style,
                {
                    "semantic_mode": "cautious_probe",
                    "suggested_style": "cautious",
                    "fused_mode": "locomotion",
                },
            )
            return {
                "ok": True,
                "active": False,
                "reason": "fallback_no_terrain_objective_rule",
                "terrain": terrain,
                "objective_name": objective_name,
                "objective_distance": objective_distance,
                "beta_norm": beta_norm,
                "selected_style": fallback_style,
                "semantic_mode": semantic.get("semantic_mode", "cautious_probe"),
                "suggested_style": semantic.get("suggested_style", "cautious"),
                "fused_mode": semantic.get("fused_mode", "locomotion"),
                "vote_confidence": 0.0,
                "vote_scores": {},
            }

        confidence = _f(rule.get("vote_confidence"), 0.0)
        active = confidence >= float(min_confidence)

        return {
            "ok": True,
            "active": active,
            "reason": "selected" if active else "below_min_confidence",
            "terrain": terrain,
            "objective_name": objective_name,
            "objective_distance": objective_distance,
            "beta_norm": beta_norm,
            "selected_style": rule.get("selected_style", "cautious"),
            "semantic_mode": rule.get("semantic_mode", "cautious_probe"),
            "suggested_style": rule.get("suggested_style", "cautious"),
            "fused_mode": rule.get("fused_mode", "locomotion"),
            "vote_confidence": confidence,
            "vote_scores": rule.get("vote_scores", {}),
        }


def apply_objective_conditioned_selection(
    entry: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Return a copied policy entry with selected semantic/style fields applied."""
    out = dict(entry)
    out["semantic_mode"] = selection.get("semantic_mode", out.get("semantic_mode", "unknown"))
    out["semantic"] = selection.get("semantic_mode", out.get("semantic", "unknown"))
    out["decision"] = selection.get("semantic_mode", out.get("decision", "unknown"))
    out["suggested_style"] = selection.get("suggested_style", out.get("suggested_style", "unknown"))
    out["fused_mode"] = selection.get("fused_mode", out.get("fused_mode", "locomotion"))
    out["objective_conditioned_selector"] = {
        "objective_name": selection.get("objective_name"),
        "selected_style": selection.get("selected_style"),
        "semantic_mode": selection.get("semantic_mode"),
        "suggested_style": selection.get("suggested_style"),
        "vote_confidence": selection.get("vote_confidence"),
        "reason": selection.get("reason"),
    }
    return out
