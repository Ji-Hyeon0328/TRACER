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


def make_x(
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


def candidate_action(row: dict[str, Any]) -> tuple[float, float, float, float]:
    # Supports aggregate CSV rows.
    if "ref_vx" in row:
        return (
            to_float(row.get("ref_vx")),
            to_float(row.get("ref_yaw_rate")),
            to_float(row.get("ref_body_height")),
            to_float(row.get("ref_swing_clearance")),
        )

    # Supports empirical manifest rows.
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
        out.append(r)

    return out


def ram_safe_score(pred: dict[str, float]) -> float:
    # Diagnostic scalar only. Higher means safer/more teacher-like.
    return (
        0.90 * pred["empirical_teacher_weight"]
        + 0.70 * pred["empirical_stable_prob_lcb95"]
        - 0.80 * pred["empirical_recovery_prob"]
        - 0.45 * pred["empirical_uncertainty"]
        - 0.45 * pred["empirical_zmin_risk"]
        - 0.35 * pred["empirical_below22_risk"]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/phase_a_ram_empirical_v1/phase_a_ram_empirical_v1.pt")
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--profile", default="balanced", choices=sorted(PROFILE_BETA))

    # Single action mode.
    ap.add_argument("--vx", type=float, default=None)
    ap.add_argument("--yaw-rate", type=float, default=0.0)
    ap.add_argument("--body-height", type=float, default=None)
    ap.add_argument("--swing-clearance", type=float, default=None)

    # Candidate ranking mode.
    ap.add_argument("--candidate-csv", default="")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    terrain_vocab = list(ckpt.get("terrain_vocab", []))
    target_schema = list(ckpt.get("target_schema", DEFAULT_TARGETS))
    input_dim = int(ckpt.get("input_dim"))
    output_dim = int(ckpt.get("output_dim", len(target_schema)))

    model = RamEmpiricalV1(input_dim, output_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    beta = PROFILE_BETA[args.profile]

    query_items: list[dict[str, Any]] = []

    if args.candidate_csv:
        rows = read_csv(Path(args.candidate_csv))
        candidates = unique_candidates(rows, args.terrain)
        if not candidates:
            raise RuntimeError(f"No candidates for terrain={args.terrain} in {args.candidate_csv}")

        for r in candidates:
            vx, yaw, h, c = candidate_action(r)
            query_items.append({
                "source": "candidate_csv",
                "terrain": args.terrain,
                "profile": args.profile,
                "vx": vx,
                "yaw_rate": yaw,
                "body_height": h,
                "swing_clearance": c,
                "aggregate_outcome_label": r.get("outcome_label", r.get("empirical_outcome_label", "")),
                "aggregate_stable_prob": to_float(r.get("stable_prob", r.get("empirical_stable_prob", ""))),
                "aggregate_lcb95": to_float(r.get("stable_prob_lcb95", r.get("empirical_stable_prob_lcb95", ""))),
                "aggregate_uncertainty": to_float(r.get("outcome_uncertainty", r.get("empirical_uncertainty", ""))),
                "aggregate_n_episodes": int(to_float(r.get("n_episodes", r.get("empirical_n_episodes", 0)))),
            })
    else:
        if args.vx is None or args.body_height is None or args.swing_clearance is None:
            raise RuntimeError(
                "Single-action mode requires --vx, --body-height, and --swing-clearance. "
                "Or provide --candidate-csv."
            )

        query_items.append({
            "source": "single_action",
            "terrain": args.terrain,
            "profile": args.profile,
            "vx": args.vx,
            "yaw_rate": args.yaw_rate,
            "body_height": args.body_height,
            "swing_clearance": args.swing_clearance,
        })

    xs = torch.tensor(
        [
            make_x(
                terrain=item["terrain"],
                terrain_vocab=terrain_vocab,
                beta=beta,
                vx=float(item["vx"]),
                yaw_rate=float(item["yaw_rate"]),
                body_height=float(item["body_height"]),
                swing_clearance=float(item["swing_clearance"]),
            )
            for item in query_items
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        preds = model(xs).detach().tolist()

    results = []
    for item, pred_vec in zip(query_items, preds):
        pred = {k: float(v) for k, v in zip(target_schema, pred_vec)}
        item_out = dict(item)
        item_out["prediction"] = pred
        item_out["ram_safe_score"] = ram_safe_score(pred)
        results.append(item_out)

    results.sort(key=lambda r: float(r["ram_safe_score"]), reverse=True)

    output = {
        "model": args.model,
        "terrain": args.terrain,
        "profile": args.profile,
        "beta": {
            "motion": beta[0],
            "stability": beta[1],
            "energy": beta[2],
        },
        "target_schema": target_schema,
        "results": results,
        "note": (
            "RAM empirical v1 query. Predictions are aggregate-style recovery/uncertainty/risk "
            "estimates conditioned on terrain, beta, and high-level action."
        ),
    }

    print(f"[TRACER] RAM empirical query terrain={args.terrain} profile={args.profile}")
    print(f"[TRACER] beta={beta}")
    print()

    print("===== RAM-ranked candidates =====")
    for i, r in enumerate(results[: args.top_k], start=1):
        p = r["prediction"]
        print(
            f"{i:02d}. score={r['ram_safe_score']:.4f} "
            f"a=[{r['vx']:.3f},{r['yaw_rate']:.3f},{r['body_height']:.3f},{r['swing_clearance']:.3f}] "
            f"rec={p['empirical_recovery_prob']:.3f} "
            f"unc={p['empirical_uncertainty']:.3f} "
            f"lcb={p['empirical_stable_prob_lcb95']:.3f} "
            f"zrisk={p['empirical_zmin_risk']:.3f} "
            f"b22={p['empirical_below22_risk']:.3f} "
            f"tw={p['empirical_teacher_weight']:.3f} "
            f"label={r.get('aggregate_outcome_label','')}"
        )

    print()
    print(json.dumps(output, indent=2))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print()
        print(f"[TRACER] wrote json: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
