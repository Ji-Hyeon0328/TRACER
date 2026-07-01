#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


PROFILE_BETA = {
    "balanced":          [0.34, 0.33, 0.33],
    "motion":            [0.65, 0.20, 0.15],
    "stability":         [0.20, 0.65, 0.15],
    "energy":            [0.20, 0.20, 0.60],
    "motion_extreme":    [0.85, 0.10, 0.05],
    "stability_extreme": [0.05, 0.90, 0.05],
    "energy_extreme":    [0.05, 0.10, 0.85],
}


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def terrain_onehot(terrain: str, vocab: list[str]) -> list[float]:
    return [1.0 if terrain == t else 0.0 for t in vocab]


class ObjectiveUtility(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def context_features(row: dict[str, Any], terrain_vocab: list[str]) -> list[float]:
    terrain = str(row.get("terrain", "unknown"))
    return [
        *terrain_onehot(terrain, terrain_vocab),

        to_float(row.get("a_hl_vx")),
        to_float(row.get("a_hl_yaw_rate")),
        to_float(row.get("a_hl_body_height")),
        to_float(row.get("a_hl_swing_clearance")),

        to_float(row.get("stable_frac")),
        to_float(row.get("zmin_mean")),
        to_float(row.get("below22_mean")),
        to_float(row.get("distance_mean")),
        to_float(row.get("rollmax_mean")),
        to_float(row.get("pitchmax_mean")),

        to_float(row.get("ram_vertical_sink_target")),
        to_float(row.get("ram_progress_mismatch_target")),
        to_float(row.get("ram_orientation_risk_target")),
        to_float(row.get("ram_low_height_risk_target")),
        to_float(row.get("ram_recovery_needed_target")),
        to_float(row.get("ram_uncertainty_proxy_target")),
    ]


def make_x(row: dict[str, Any], terrain_vocab: list[str], beta: list[float]) -> list[float]:
    return [
        *context_features(row, terrain_vocab),
        beta[0],
        beta[1],
        beta[2],
    ]


def filter_context_rows(
    rows: list[dict[str, Any]],
    terrain: str,
    preset: str,
    label_filter: str,
) -> list[dict[str, Any]]:
    out = [r for r in rows if str(r.get("terrain", "")) == terrain]

    if preset:
        out = [r for r in out if str(r.get("preset", "")) == preset]

    if label_filter:
        allowed = {x.strip() for x in label_filter.split(",") if x.strip()}
        out = [r for r in out if str(r.get("manifest_label", "")) in allowed]

    return out


def representative_row(rows: list[dict[str, Any]], terrain: str, preset: str, label_filter: str) -> dict[str, Any]:
    candidates = filter_context_rows(rows, terrain, preset, label_filter)
    if not candidates:
        raise RuntimeError(
            f"No context rows for terrain={terrain}, preset={preset!r}, label_filter={label_filter!r}"
        )

    # Prefer lower recovery/uncertainty and higher score.
    candidates.sort(
        key=lambda r: (
            to_float(r.get("ram_recovery_needed_target")),
            to_float(r.get("ram_uncertainty_proxy_target")),
            -to_float(r.get("R_score")),
        )
    )
    return candidates[0]


def average_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No rows for average context")

    numeric_keys = [
        "a_hl_vx",
        "a_hl_yaw_rate",
        "a_hl_body_height",
        "a_hl_swing_clearance",
        "stable_frac",
        "zmin_mean",
        "below22_mean",
        "distance_mean",
        "rollmax_mean",
        "pitchmax_mean",
        "ram_vertical_sink_target",
        "ram_progress_mismatch_target",
        "ram_orientation_risk_target",
        "ram_low_height_risk_target",
        "ram_recovery_needed_target",
        "ram_uncertainty_proxy_target",
        "R_score",
    ]

    out: dict[str, Any] = {
        "terrain": rows[0].get("terrain", ""),
        "actual_world_name": rows[0].get("actual_world_name", ""),
        "preset": "AVERAGED_CONTEXT",
        "manifest_label": "averaged_context",
    }

    for k in numeric_keys:
        out[k] = sum(to_float(r.get(k)) for r in rows) / max(1, len(rows))

    return out


def normalize(vals: list[float]) -> list[float]:
    lo = min(vals)
    hi = max(vals)
    if abs(hi - lo) < 1e-9:
        return [0.5 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def beta_similarity(beta: list[float], target: list[float]) -> float:
    # Both are simplex-like. Max L1 distance is about 2.
    return max(0.0, 1.0 - min(2.0, sum(abs(a - b) for a, b in zip(beta, target))) / 2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="artifacts/phase_a_objective_selector_pref_v0/phase_a_objective_selector_pref_v0.pt",
    )
    ap.add_argument(
        "--manifest-csv",
        default="data/phase_a_training_manifest_v0/phase_a_training_manifest_v0.csv",
    )
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--preset", default="")
    ap.add_argument("--label-filter", default="positive_teacher,borderline_teacher")
    ap.add_argument("--context-mode", default="best", choices=["best", "average"])

    ap.add_argument(
        "--mission-profile",
        default="balanced",
        choices=sorted(PROFILE_BETA),
        help="Mission-level objective prior. This prevents the selector from always collapsing to stability_extreme.",
    )
    ap.add_argument("--mission-prior-strength", type=float, default=0.8)
    ap.add_argument("--safety-prior-strength", type=float, default=0.15)
    ap.add_argument("--recovery-prior-strength", type=float, default=1.0)
    ap.add_argument("--recovery-energy-penalty", type=float, default=0.8)
    ap.add_argument("--recovery-motion-penalty", type=float, default=0.6)

    ap.add_argument("--recovery-threshold", type=float, default=0.5)
    ap.add_argument("--low-height-risk-threshold", type=float, default=0.55)
    ap.add_argument("--vertical-sink-threshold", type=float, default=0.50)
    ap.add_argument("--force-recovery-stability", action="store_true")

    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    terrain_vocab = list(ckpt.get("terrain_vocab", []))
    input_dim = int(ckpt.get("input_dim"))

    model = ObjectiveUtility(in_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = read_csv(Path(args.manifest_csv))
    matching = filter_context_rows(rows, args.terrain, args.preset, args.label_filter)

    if args.context_mode == "average":
        ctx = average_context(matching)
    else:
        ctx = representative_row(rows, args.terrain, args.preset, args.label_filter)

    names = list(PROFILE_BETA.keys())
    betas = [PROFILE_BETA[n] for n in names]
    xs = torch.tensor([make_x(ctx, terrain_vocab, b) for b in betas], dtype=torch.float32)

    with torch.no_grad():
        raw_utils = model(xs).detach().tolist()

    norm_utils = normalize(raw_utils)
    mission_beta = PROFILE_BETA[args.mission_profile]

    recovery_needed = to_float(ctx.get("ram_recovery_needed_target"))
    low_height_risk = to_float(ctx.get("ram_low_height_risk_target"))
    vertical_sink = to_float(ctx.get("ram_vertical_sink_target"))
    uncertainty = to_float(ctx.get("ram_uncertainty_proxy_target"))

    recovery_gate = (
        recovery_needed >= args.recovery_threshold
        or low_height_risk >= args.low_height_risk_threshold
        or vertical_sink >= args.vertical_sink_threshold
    )

    ranked = []
    for name, beta, raw_u, norm_u in zip(names, betas, raw_utils, norm_utils):
        mission_score = beta_similarity(beta, mission_beta)
        safety_score = beta[1]

        adjusted = (
            norm_u
            + args.mission_prior_strength * mission_score
            + args.safety_prior_strength * safety_score
        )

        if recovery_gate:
            adjusted += (
                args.recovery_prior_strength * beta[1]
                - args.recovery_energy_penalty * beta[2]
                - args.recovery_motion_penalty * beta[0]
            )

        if args.force_recovery_stability and recovery_gate and name not in {"stability", "stability_extreme"}:
            adjusted -= 10.0

        ranked.append({
            "profile": name,
            "beta_motion": beta[0],
            "beta_stability": beta[1],
            "beta_energy": beta[2],
            "learned_utility_raw": raw_u,
            "learned_utility_norm": norm_u,
            "mission_score": mission_score,
            "safety_score": safety_score,
            "adjusted_score": adjusted,
        })

    ranked.sort(key=lambda r: float(r["adjusted_score"]), reverse=True)
    best = ranked[0]

    result = {
        "terrain": args.terrain,
        "context_mode": args.context_mode,
        "preset_filter": args.preset,
        "label_filter": args.label_filter,
        "mission_profile": args.mission_profile,
        "recovery_gate": recovery_gate,
        "guard_reason": {
            "recovery_needed": recovery_needed,
            "low_height_risk": low_height_risk,
            "vertical_sink": vertical_sink,
            "uncertainty_proxy": uncertainty,
            "recovery_threshold": args.recovery_threshold,
            "low_height_risk_threshold": args.low_height_risk_threshold,
            "vertical_sink_threshold": args.vertical_sink_threshold,
        },
        "context": {
            "preset": ctx.get("preset", ""),
            "manifest_label": ctx.get("manifest_label", ""),
            "a_hl_vx": to_float(ctx.get("a_hl_vx")),
            "a_hl_yaw_rate": to_float(ctx.get("a_hl_yaw_rate")),
            "a_hl_body_height": to_float(ctx.get("a_hl_body_height")),
            "a_hl_swing_clearance": to_float(ctx.get("a_hl_swing_clearance")),
            "stable_frac": to_float(ctx.get("stable_frac")),
            "zmin_mean": to_float(ctx.get("zmin_mean")),
            "below22_mean": to_float(ctx.get("below22_mean")),
            "ram_vertical_sink_target": vertical_sink,
            "ram_progress_mismatch_target": to_float(ctx.get("ram_progress_mismatch_target")),
            "ram_orientation_risk_target": to_float(ctx.get("ram_orientation_risk_target")),
            "ram_low_height_risk_target": low_height_risk,
            "ram_recovery_needed_target": recovery_needed,
            "ram_uncertainty_proxy_target": uncertainty,
        },
        "selected": best,
        "ranked_profiles": ranked,
        "note": (
            "Guarded Phase-A Objective Selector query. Uses learned utility plus mission prior "
            "and recovery/safety guard. This is closer to TRACER runtime logic than raw preference utility."
        ),
    }

    print(f"[TRACER] terrain={args.terrain}")
    print(f"[TRACER] context_mode={args.context_mode}")
    print(f"[TRACER] mission_profile={args.mission_profile}")
    print(f"[TRACER] recovery_gate={recovery_gate}")
    print(f"[TRACER] context preset={ctx.get('preset','')} label={ctx.get('manifest_label','')}")
    print()

    print("===== guarded ranked beta profiles =====")
    for i, r in enumerate(ranked, start=1):
        print(
            f"{i:02d}. adjusted={float(r['adjusted_score']):.4f} "
            f"raw={float(r['learned_utility_raw']):.4f} "
            f"norm={float(r['learned_utility_norm']):.4f} "
            f"profile={r['profile']:18s} "
            f"beta=[{r['beta_motion']:.2f},{r['beta_stability']:.2f},{r['beta_energy']:.2f}] "
            f"mission={float(r['mission_score']):.3f} safety={float(r['safety_score']):.3f}"
        )

    print()
    print(json.dumps(result, indent=2))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print()
        print(f"[TRACER] wrote json: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
