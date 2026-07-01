#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DEFAULT_TARGETS = [
    "empirical_recovery_prob",
    "empirical_uncertainty",
    "empirical_stable_prob_lcb95",
    "empirical_zmin_risk",
    "empirical_below22_risk",
    "empirical_teacher_weight",
]


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


class RamEmpiricalV1(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def candidate_action(row: dict[str, Any]) -> tuple[float, float, float, float]:
    if "ref_vx" in row:
        return (
            to_float(row.get("ref_vx")),
            to_float(row.get("ref_yaw_rate")),
            to_float(row.get("ref_body_height")),
            to_float(row.get("ref_swing_clearance")),
        )

    return (
        to_float(row.get("a_hl_vx")),
        to_float(row.get("a_hl_yaw_rate")),
        to_float(row.get("a_hl_body_height")),
        to_float(row.get("a_hl_swing_clearance")),
    )


def unique_candidates(rows: list[dict[str, Any]], terrain: str) -> list[dict[str, Any]]:
    out = []
    seen = set()

    for r in rows:
        if str(r.get("terrain", "")) != terrain:
            continue

        vx, yaw, h, c = candidate_action(r)
        key = (
            terrain,
            round(vx, 6),
            round(yaw, 6),
            round(h, 6),
            round(c, 6),
        )
        if key in seen:
            continue
        seen.add(key)

        rr = dict(r)
        rr["_vx"] = vx
        rr["_yaw_rate"] = yaw
        rr["_body_height"] = h
        rr["_swing_clearance"] = c
        out.append(rr)

    return out


def make_x(
    *,
    terrain: str,
    terrain_vocab: list[str],
    beta: list[float],
    vx: float,
    yaw_rate: float,
    body_height: float,
    swing_clearance: float,
) -> list[float]:
    return [
        *terrain_onehot(terrain, terrain_vocab),
        beta[0],
        beta[1],
        beta[2],
        vx,
        yaw_rate,
        body_height,
        swing_clearance,
    ]


def ram_safe_score(pred: dict[str, float]) -> float:
    return (
        0.90 * pred["empirical_teacher_weight"]
        + 0.70 * pred["empirical_stable_prob_lcb95"]
        - 0.80 * pred["empirical_recovery_prob"]
        - 0.45 * pred["empirical_uncertainty"]
        - 0.45 * pred["empirical_zmin_risk"]
        - 0.35 * pred["empirical_below22_risk"]
    )


def pass_ram_guard(
    pred: dict[str, float],
    *,
    max_recovery: float,
    max_uncertainty: float,
    min_lcb95: float,
    max_zmin_risk: float,
    max_below22_risk: float,
    min_teacher_weight: float,
) -> bool:
    return (
        pred["empirical_recovery_prob"] <= max_recovery
        and pred["empirical_uncertainty"] <= max_uncertainty
        and pred["empirical_stable_prob_lcb95"] >= min_lcb95
        and pred["empirical_zmin_risk"] <= max_zmin_risk
        and pred["empirical_below22_risk"] <= max_below22_risk
        and pred["empirical_teacher_weight"] >= min_teacher_weight
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--objective-json", required=True)
    ap.add_argument(
        "--ram-model",
        default="artifacts/phase_a_ram_empirical_v1/phase_a_ram_empirical_v1.pt",
    )
    ap.add_argument(
        "--candidate-csv",
        default="data/phase_a_action_outcome_aggregates_v1_evidence/phase_a_action_outcome_aggregates_v1_evidence.csv",
    )

    ap.add_argument("--max-recovery", type=float, default=0.35)
    ap.add_argument("--max-uncertainty", type=float, default=0.65)
    ap.add_argument("--min-lcb95", type=float, default=0.30)
    ap.add_argument("--max-zmin-risk", type=float, default=0.55)
    ap.add_argument("--max-below22-risk", type=float, default=0.45)
    ap.add_argument("--min-teacher-weight", type=float, default=0.10)

    ap.add_argument("--preset-name", default="")
    ap.add_argument("--top-k", type=int, default=20)
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

    ckpt = torch.load(args.ram_model, map_location="cpu")
    terrain_vocab = list(ckpt.get("terrain_vocab", []))
    target_schema = list(ckpt.get("target_schema", DEFAULT_TARGETS))
    input_dim = int(ckpt.get("input_dim"))
    output_dim = int(ckpt.get("output_dim", len(target_schema)))

    model = RamEmpiricalV1(input_dim, output_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = read_csv(Path(args.candidate_csv))
    candidates = unique_candidates(rows, args.terrain)
    if not candidates:
        raise RuntimeError(f"No candidates for terrain={args.terrain}")

    xs = torch.tensor(
        [
            make_x(
                terrain=args.terrain,
                terrain_vocab=terrain_vocab,
                beta=beta,
                vx=float(r["_vx"]),
                yaw_rate=float(r["_yaw_rate"]),
                body_height=float(r["_body_height"]),
                swing_clearance=float(r["_swing_clearance"]),
            )
            for r in candidates
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        preds = model(xs).detach().tolist()

    ranked = []
    for r, pred_vec in zip(candidates, preds):
        pred = {k: float(v) for k, v in zip(target_schema, pred_vec)}
        score = ram_safe_score(pred)

        guard_pass = pass_ram_guard(
            pred,
            max_recovery=args.max_recovery,
            max_uncertainty=args.max_uncertainty,
            min_lcb95=args.min_lcb95,
            max_zmin_risk=args.max_zmin_risk,
            max_below22_risk=args.max_below22_risk,
            min_teacher_weight=args.min_teacher_weight,
        )

        ranked.append({
            "vx": float(r["_vx"]),
            "yaw_rate": float(r["_yaw_rate"]),
            "body_height": float(r["_body_height"]),
            "swing_clearance": float(r["_swing_clearance"]),
            "ram_prediction": pred,
            "ram_safe_score": score,
            "ram_guard_pass": guard_pass,
            "aggregate_outcome_label": r.get("outcome_label", r.get("empirical_outcome_label", "")),
            "aggregate_stable_prob": to_float(r.get("stable_prob", r.get("empirical_stable_prob", ""))),
            "aggregate_lcb95": to_float(r.get("stable_prob_lcb95", r.get("empirical_stable_prob_lcb95", ""))),
            "aggregate_uncertainty": to_float(r.get("outcome_uncertainty", r.get("empirical_uncertainty", ""))),
            "aggregate_n_episodes": int(to_float(r.get("n_episodes", r.get("empirical_n_episodes", 0)))),
            "source_presets_json": r.get("source_presets_json", "[]"),
        })

    ranked.sort(key=lambda r: float(r["ram_safe_score"]), reverse=True)

    safe_ranked = [r for r in ranked if r["ram_guard_pass"]]
    if safe_ranked:
        selected = safe_ranked[0]
        selection_status = "ram_guard_pass"
    else:
        selected = ranked[0]
        selection_status = "fallback_no_ram_safe_candidate"

    preset_name = args.preset_name or f"phase_a_stack_ram_{args.terrain}_{objective_profile}_v2"

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
        "ram_model": args.ram_model,
        "candidate_csv": args.candidate_csv,
        "selection_status": selection_status,
        "ram_guard": {
            "max_recovery": args.max_recovery,
            "max_uncertainty": args.max_uncertainty,
            "min_lcb95": args.min_lcb95,
            "max_zmin_risk": args.max_zmin_risk,
            "max_below22_risk": args.max_below22_risk,
            "min_teacher_weight": args.min_teacher_weight,
        },
        "selected_action": {
            "preset": preset_name,
            "vx": selected["vx"],
            "yaw_rate": selected["yaw_rate"],
            "body_height": selected["body_height"],
            "swing_clearance": selected["swing_clearance"],
            "enable": 1.0,
            "ram_safe_score": selected["ram_safe_score"],
            "ram_guard_pass": selected["ram_guard_pass"],
            "ram_prediction": selected["ram_prediction"],
            "aggregate_outcome_label": selected["aggregate_outcome_label"],
            "aggregate_stable_prob": selected["aggregate_stable_prob"],
            "aggregate_lcb95": selected["aggregate_lcb95"],
            "aggregate_uncertainty": selected["aggregate_uncertainty"],
            "aggregate_n_episodes": selected["aggregate_n_episodes"],
        },
        "ranked_candidates": ranked[: args.top_k],
        "note": (
            "RAM-aware Phase-A high-level stack export. Selects high-level action using "
            "guarded objective beta and empirical RAM predictions. This is closer to runtime "
            "TRACER logic than aggregate-ground-truth selection."
        ),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    yaml_text = f"""version: phase_a_highlevel_stack_ram_v2
description: >
  Exported by guarded Objective Selector + empirical RAM-aware action selection.
  This is closer to runtime TRACER logic than direct aggregate lookup.

terrains:
  - {args.terrain}

presets:
  - name: {preset_name}
    vx: {selected['vx']:.6f}
    yaw_rate: {selected['yaw_rate']:.6f}
    body_height: {selected['body_height']:.6f}
    swing_clearance: {selected['swing_clearance']:.6f}
    enable: 1.0
"""
    out_yaml = Path(args.out_rollout_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml_text, encoding="utf-8")

    p = selected["ram_prediction"]

    print(f"[TRACER] terrain={args.terrain}")
    print(f"[TRACER] objective_profile={objective_profile} beta={beta}")
    print(f"[TRACER] selection_status={selection_status}")
    print(
        f"[TRACER] selected: {preset_name} "
        f"a=[{selected['vx']:.3f},{selected['yaw_rate']:.3f},{selected['body_height']:.3f},{selected['swing_clearance']:.3f}] "
        f"score={selected['ram_safe_score']:.4f} "
        f"guard={selected['ram_guard_pass']} "
        f"rec={p['empirical_recovery_prob']:.3f} "
        f"unc={p['empirical_uncertainty']:.3f} "
        f"lcb={p['empirical_stable_prob_lcb95']:.3f} "
        f"zrisk={p['empirical_zmin_risk']:.3f} "
        f"b22={p['empirical_below22_risk']:.3f} "
        f"tw={p['empirical_teacher_weight']:.3f}"
    )

    print()
    print("===== RAM-ranked candidates =====")
    for i, r in enumerate(ranked[: args.top_k], start=1):
        p = r["ram_prediction"]
        print(
            f"{i:02d}. score={r['ram_safe_score']:.4f} "
            f"guard={str(r['ram_guard_pass']):5s} "
            f"a=[{r['vx']:.3f},{r['yaw_rate']:.3f},{r['body_height']:.3f},{r['swing_clearance']:.3f}] "
            f"rec={p['empirical_recovery_prob']:.3f} "
            f"unc={p['empirical_uncertainty']:.3f} "
            f"lcb={p['empirical_stable_prob_lcb95']:.3f} "
            f"zrisk={p['empirical_zmin_risk']:.3f} "
            f"b22={p['empirical_below22_risk']:.3f} "
            f"tw={p['empirical_teacher_weight']:.3f} "
            f"label={r['aggregate_outcome_label']}"
        )

    print()
    print(f"[TRACER] wrote json: {out_json}")
    print(f"[TRACER] wrote yaml: {out_yaml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
