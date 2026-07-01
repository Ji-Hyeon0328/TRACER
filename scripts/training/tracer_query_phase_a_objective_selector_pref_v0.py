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

    # Prefer high score, low recovery, lower uncertainty.
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

    keep_str = [
        "terrain",
        "actual_world_name",
        "preset",
        "manifest_label",
    ]

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

    out: dict[str, Any] = {}
    for k in keep_str:
        out[k] = rows[0].get(k, "")

    out["preset"] = "AVERAGED_CONTEXT"
    out["manifest_label"] = "averaged_context"

    for k in numeric_keys:
        out[k] = sum(to_float(r.get(k)) for r in rows) / max(1, len(rows))

    return out


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
    ap.add_argument(
        "--label-filter",
        default="positive_teacher,borderline_teacher",
        help="Comma-separated manifest labels used to select representative context.",
    )
    ap.add_argument(
        "--context-mode",
        default="best",
        choices=["best", "average"],
        help="Use best representative row or average context over matching rows.",
    )
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

    xs = []
    names = []
    betas = []
    for name, beta in PROFILE_BETA.items():
        xs.append(make_x(ctx, terrain_vocab, beta))
        names.append(name)
        betas.append(beta)

    x = torch.tensor(xs, dtype=torch.float32)

    with torch.no_grad():
        utilities = model(x).detach().tolist()

    ranked = sorted(zip(names, betas, utilities), key=lambda z: z[2], reverse=True)
    best_name, best_beta, best_utility = ranked[0]

    result = {
        "terrain": args.terrain,
        "context_mode": args.context_mode,
        "preset_filter": args.preset,
        "label_filter": args.label_filter,
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
            "ram_vertical_sink_target": to_float(ctx.get("ram_vertical_sink_target")),
            "ram_progress_mismatch_target": to_float(ctx.get("ram_progress_mismatch_target")),
            "ram_orientation_risk_target": to_float(ctx.get("ram_orientation_risk_target")),
            "ram_low_height_risk_target": to_float(ctx.get("ram_low_height_risk_target")),
            "ram_recovery_needed_target": to_float(ctx.get("ram_recovery_needed_target")),
            "ram_uncertainty_proxy_target": to_float(ctx.get("ram_uncertainty_proxy_target")),
        },
        "selected": {
            "profile": best_name,
            "beta_motion": best_beta[0],
            "beta_stability": best_beta[1],
            "beta_energy": best_beta[2],
            "utility": best_utility,
        },
        "ranked_profiles": [
            {
                "profile": n,
                "beta_motion": b[0],
                "beta_stability": b[1],
                "beta_energy": b[2],
                "utility": u,
            }
            for n, b, u in ranked
        ],
        "note": (
            "Phase-A bootstrapped Objective Selector query. This chooses beta under "
            "a manifest/RAM-derived context. It is not yet the final online Objective Selector."
        ),
    }

    print(f"[TRACER] terrain={args.terrain}")
    print(f"[TRACER] context_mode={args.context_mode}")
    print(f"[TRACER] context preset={ctx.get('preset','')} label={ctx.get('manifest_label','')}")
    print()

    print("===== ranked beta profiles =====")
    for i, (n, b, u) in enumerate(ranked, start=1):
        print(
            f"{i:02d}. utility={u:.4f} "
            f"profile={n:18s} "
            f"beta=[{b[0]:.2f},{b[1]:.2f},{b[2]:.2f}]"
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
