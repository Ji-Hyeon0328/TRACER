#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


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


class TerrainAwareRanker(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
            nn.Linear(96, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def unique_candidates(rows: list[dict[str, Any]], terrain: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()

    for r in rows:
        if str(r.get("terrain", "")) != terrain:
            continue

        key = (
            terrain,
            str(r.get("preset", "")),
            round(to_float(r.get("ref_vx")), 6),
            round(to_float(r.get("ref_yaw_rate")), 6),
            round(to_float(r.get("ref_body_height")), 6),
            round(to_float(r.get("ref_swing_clearance")), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    return out


def get_train_score(row: dict[str, Any]) -> float:
    if "R_train_score" in row:
        return to_float(row.get("R_train_score"))
    if "R_profile_v2_gated_mean" in row:
        return to_float(row.get("R_profile_v2_gated_mean"))
    return to_float(row.get("R_gated_mean"))


def label_bonus(label: str) -> float:
    if label == "positive_teacher":
        return 0.04
    if label == "borderline":
        return 0.02
    if label == "risk_negative":
        return -0.08
    return 0.0


def selection_score(model_score: float, row: dict[str, Any], mode: str) -> float:
    bank_score = get_train_score(row)

    if mode == "model":
        return float(model_score)
    if mode == "bank":
        return bank_score + label_bonus(str(row.get("bank_label", "")))

    return (
        0.55 * float(model_score)
        + 0.45 * bank_score
        + label_bonus(str(row.get("bank_label", "")))
    )


def make_ranker_x(beta: list[float], terrain: str, row: dict[str, Any], terrain_vocab: list[str]) -> list[float]:
    return [
        beta[0],
        beta[1],
        beta[2],
        *terrain_onehot(terrain, terrain_vocab),
        to_float(row.get("ref_vx")),
        to_float(row.get("ref_yaw_rate")),
        to_float(row.get("ref_body_height")),
        to_float(row.get("ref_swing_clearance")),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--objective-json", required=True)
    ap.add_argument(
        "--ranker-model",
        default="artifacts/phase_a_terrain_aware_ranker_v0/phase_a_terrain_aware_ranker_v0.pt",
    )
    ap.add_argument(
        "--candidate-csv",
        default="data/phase_a_terrain_aware_bank_v0/phase_a_terrain_aware_candidates_v0.csv",
    )
    ap.add_argument("--selection-mode", default="hybrid", choices=["model", "bank", "hybrid"])
    ap.add_argument("--allow-risk", action="store_true")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--preset-name", default="")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-rollout-yaml", required=True)
    args = ap.parse_args()

    objective = json.loads(Path(args.objective_json).read_text(encoding="utf-8"))
    selected_obj = objective["selected"]

    beta = [
        to_float(selected_obj.get("beta_motion")),
        to_float(selected_obj.get("beta_stability")),
        to_float(selected_obj.get("beta_energy")),
    ]
    objective_profile = str(selected_obj.get("profile", "custom_beta"))

    ckpt = torch.load(args.ranker_model, map_location="cpu")
    terrain_vocab = list(ckpt.get("terrain_vocab", []))
    input_dim = int(ckpt.get("input_dim"))

    model = TerrainAwareRanker(in_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = read_csv(Path(args.candidate_csv))
    candidates = unique_candidates(rows, args.terrain)
    if not candidates:
        raise RuntimeError(f"No ranker candidates for terrain={args.terrain}")

    xs = torch.tensor(
        [make_ranker_x(beta, args.terrain, r, terrain_vocab) for r in candidates],
        dtype=torch.float32,
    )

    with torch.no_grad():
        pred_scores = model(xs).detach().tolist()

    raw_ranked = []
    for r, pred_s in zip(candidates, pred_scores):
        sel_s = selection_score(float(pred_s), r, args.selection_mode)
        raw_ranked.append((r, float(pred_s), float(sel_s)))

    selectable = raw_ranked
    if not args.allow_risk:
        non_risk = [x for x in raw_ranked if str(x[0].get("bank_label", "")) != "risk_negative"]
        if non_risk:
            selectable = non_risk

    ranked = sorted(selectable, key=lambda x: x[2], reverse=True)
    display_ranked = sorted(raw_ranked, key=lambda x: x[2], reverse=True)

    best, best_pred, best_select = ranked[0]

    vx = to_float(best.get("ref_vx"))
    yaw = to_float(best.get("ref_yaw_rate"))
    h = to_float(best.get("ref_body_height"))
    c = to_float(best.get("ref_swing_clearance"))

    preset_name = args.preset_name or f"phase_a_stack_{args.terrain}_{objective_profile}_v0"

    result = {
        "terrain": args.terrain,
        "objective_json": args.objective_json,
        "objective_selected": selected_obj,
        "objective_profile": objective_profile,
        "beta": {
            "motion": beta[0],
            "stability": beta[1],
            "energy": beta[2],
        },
        "ranker": {
            "model": args.ranker_model,
            "candidate_csv": args.candidate_csv,
            "selection_mode": args.selection_mode,
            "allow_risk": args.allow_risk,
        },
        "selected_action": {
            "preset": preset_name,
            "source_candidate": best.get("preset", ""),
            "bank_label": best.get("bank_label", ""),
            "model_score": best_pred,
            "selection_score": best_select,
            "bank_train_score": get_train_score(best),
            "vx": vx,
            "yaw_rate": yaw,
            "body_height": h,
            "swing_clearance": c,
            "enable": 1.0,
        },
        "ranked_candidates": [
            {
                "preset": r.get("preset", ""),
                "bank_label": r.get("bank_label", ""),
                "model_score": pred_s,
                "selection_score": sel_s,
                "bank_train_score": get_train_score(r),
                "vx": to_float(r.get("ref_vx")),
                "yaw_rate": to_float(r.get("ref_yaw_rate")),
                "body_height": to_float(r.get("ref_body_height")),
                "swing_clearance": to_float(r.get("ref_swing_clearance")),
            }
            for r, pred_s, sel_s in display_ranked[: args.top_k]
        ],
        "note": (
            "Phase-A high-level stack export: guarded objective beta plus terrain-aware "
            "ranker-selected high-level reference. This is a scaffold for TRACER meta planner."
        ),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    yaml_text = f"""version: phase_a_highlevel_stack_v0
description: >
  Exported by guarded Objective Selector + terrain-aware candidate ranker.
  This is a Phase-A high-level stack reference, not the final RL meta-gait policy.

terrains:
  - {args.terrain}

presets:
  - name: {preset_name}
    vx: {vx:.6f}
    yaw_rate: {yaw:.6f}
    body_height: {h:.6f}
    swing_clearance: {c:.6f}
    enable: 1.0
"""
    out_yaml = Path(args.out_rollout_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml_text, encoding="utf-8")

    print(f"[TRACER] terrain={args.terrain}")
    print(f"[TRACER] objective_profile={objective_profile} beta={beta}")
    print(f"[TRACER] selected action: {preset_name}")
    print(
        f"  source={best.get('preset','')} label={best.get('bank_label','')} "
        f"vx={vx:.3f} yaw={yaw:.3f} h={h:.3f} c={c:.3f} "
        f"model={best_pred:.4f} select={best_select:.4f} bank={get_train_score(best):.4f}"
    )
    print()
    print("===== ranked candidates =====")
    for i, item in enumerate(result["ranked_candidates"], start=1):
        print(
            f"{i:02d}. select={item['selection_score']:.4f} "
            f"model={item['model_score']:.4f} "
            f"bank={item['bank_train_score']:.4f} "
            f"label={item['bank_label']:16s} "
            f"preset={item['preset']:40s} "
            f"a=[{item['vx']:.3f},{item['body_height']:.3f},{item['swing_clearance']:.3f}]"
        )

    print()
    print(f"[TRACER] wrote json: {out_json}")
    print(f"[TRACER] wrote yaml: {out_yaml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
