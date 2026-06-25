from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tracer_core.highlevel.gait_mode_selector import input_from_policy_entry, select_gait_mode
from tracer_core.highlevel.meta_gait import HighLevelPolicyInput, MetaGaitCommand
from tracer_core.highlevel.meta_gait_policy import MetaGaitPolicy, make_meta_gait_policy
from tracer_core.highlevel.policy_input_builder import (
    build_high_level_policy_input,
    high_level_policy_input_summary,
)


MODE_ORDER = ("fast", "normal", "conservative", "recovery", "avoid", "no_valid")
RAM_LEVEL_ORDER = ("unknown", "stable", "caution", "unstable")


FEATURE_NAMES = (
    "beta_motion",
    "beta_stability",
    "beta_energy",
    "sigma",
    "control_risk",
    "fallen_prob",
    "recovery_prob",
    "rho_norm_feature",
    "ram_level_unknown",
    "ram_level_stable",
    "ram_level_caution",
    "ram_level_unstable",
    "mode_fast",
    "mode_normal",
    "mode_conservative",
    "mode_recovery",
    "mode_avoid",
    "mode_no_valid",
)


TARGET_NAMES = (
    "vx",
    "yaw_rate",
    "body_height",
    "swing_clearance",
    "enable",
    "gait_period",
    "duty_factor",
    "step_length",
    "stance_width",
    "impedance_scale",
    "residual_gain_scale",
    "risk_scale",
)


def _one_hot(value: str, order: Sequence[str]) -> list[float]:
    return [1.0 if value == k else 0.0 for k in order]


def _safe_first(xs: Sequence[float]) -> float:
    return float(xs[0]) if len(xs) > 0 else 0.0


def policy_input_to_feature_vector(inp: HighLevelPolicyInput) -> list[float]:
    """Flatten HighLevelPolicyInput into a small numeric vector for v0 training.

    This is intentionally compact. Full context c_t, full rho, robot_state, and goal
    will be added later once those runtime streams are finalized.
    """

    return [
        float(inp.beta.motion),
        float(inp.beta.stability),
        float(inp.beta.energy),
        float(inp.ram.sigma),
        float(inp.ram.control_risk),
        float(inp.ram.fallen_prob),
        float(inp.ram.recovery_prob),
        _safe_first(inp.ram.rho),
        *_one_hot(inp.ram.ram_level, RAM_LEVEL_ORDER),
        *_one_hot(inp.gait_mode, MODE_ORDER),
    ]


def meta_gait_to_target_vector(meta: MetaGaitCommand) -> list[float]:
    return [
        float(meta.vx),
        float(meta.yaw_rate),
        float(meta.body_height),
        float(meta.swing_clearance),
        float(meta.enable),
        float(meta.gait_period),
        float(meta.duty_factor),
        float(meta.step_length),
        float(meta.stance_width),
        float(meta.impedance_scale),
        float(meta.residual_gain_scale),
        float(meta.risk_scale),
    ]


@dataclass(frozen=True)
class MetaGaitDatasetSample:
    terrain_key: str
    policy_input: HighLevelPolicyInput
    target_meta_gait: MetaGaitCommand
    feature_vector: tuple[float, ...]
    target_vector: tuple[float, ...]
    policy_input_summary: Mapping[str, Any]
    source: str = "rule_based_meta_gait_policy"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "terrain_key": self.terrain_key,
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "feature_vector": list(self.feature_vector),
            "target_vector": list(self.target_vector),
            "policy_input": asdict(self.policy_input),
            "target_meta_gait": asdict(self.target_meta_gait),
            "policy_input_summary": dict(self.policy_input_summary),
            "source": self.source,
        }


def build_meta_gait_sample_from_policy_entry(
    terrain_key: str,
    policy_entry: Mapping[str, Any],
    *,
    policy: MetaGaitPolicy | None = None,
    ram_level: str = "unknown",
    ram_gate_action: str = "unknown",
    gate: Mapping[str, Any] | None = None,
) -> MetaGaitDatasetSample:
    """Build one supervised sample x -> theta_t from a fusion-policy entry."""

    if policy is None:
        policy = make_meta_gait_policy("rule_based")

    gms_in = input_from_policy_entry(
        policy_entry,
        ram_level=ram_level,
        ram_gate_action=ram_gate_action,
        sigma_mean=gate.get("sigma_mean") if gate else None,
        control_risk=gate.get("control_risk", 0.0) if gate else 0.0,
        fallen_prob=gate.get("fallen_prob", 0.0) if gate else 0.0,
        recovery_prob=gate.get("recovery_prob", 0.0) if gate else 0.0,
    )
    gms_out = select_gait_mode(gms_in)

    policy_input = build_high_level_policy_input(
        policy_entry=policy_entry,
        gms_in=gms_in,
        gms_out=gms_out,
        gate=gate,
        context=[],
        robot_state=[],
        goal=[],
    )
    meta = policy.predict(policy_input)

    return MetaGaitDatasetSample(
        terrain_key=terrain_key,
        policy_input=policy_input,
        target_meta_gait=meta,
        feature_vector=tuple(policy_input_to_feature_vector(policy_input)),
        target_vector=tuple(meta_gait_to_target_vector(meta)),
        policy_input_summary=high_level_policy_input_summary(policy_input),
    )


def _terrain_entries_from_policy_json(data: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(data.get("terrains"), Mapping):
        return data["terrains"]  # type: ignore[return-value]

    # Fallback: support a plain {terrain_key: entry} dictionary.
    candidate = {
        k: v
        for k, v in data.items()
        if isinstance(v, Mapping) and isinstance(v.get("command"), Mapping)
    }
    return candidate


def build_meta_gait_samples_from_policy_json(
    policy_json: str | Path,
    *,
    terrain_keys: Iterable[str] | None = None,
    policy: MetaGaitPolicy | None = None,
) -> list[MetaGaitDatasetSample]:
    path = Path(policy_json).expanduser()
    data = json.loads(path.read_text())
    entries = _terrain_entries_from_policy_json(data)

    selected = set(terrain_keys) if terrain_keys is not None else None
    samples: list[MetaGaitDatasetSample] = []

    for terrain_key, entry in sorted(entries.items()):
        if selected is not None and terrain_key not in selected:
            continue
        if not isinstance(entry, Mapping) or not isinstance(entry.get("command"), Mapping):
            continue
        samples.append(
            build_meta_gait_sample_from_policy_entry(
                terrain_key,
                entry,
                policy=policy,
            )
        )

    return samples


def write_jsonl(samples: Sequence[MetaGaitDatasetSample], output_path: str | Path) -> Path:
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_json_dict(), sort_keys=True) + "\n")
    return out
