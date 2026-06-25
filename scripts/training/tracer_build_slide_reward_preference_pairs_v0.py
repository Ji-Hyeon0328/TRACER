#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
    except Exception:
        pass
    return default


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    priority = [
        "pair_id",
        "reason",
        "pair_type",
        "confidence",
        "winner_reward",
        "loser_reward",
        "reward_gap",
        "winner_tag",
        "winner_style",
        "winner_primitive_family",
        "winner_fall_like",
        "loser_tag",
        "loser_style",
        "loser_primitive_family",
        "loser_fall_like",
    ]

    fields = priority + sorted({k for r in rows for k in r.keys()} - set(priority))

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def prefixed(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in row.items()}


def pair_reason(w: dict[str, Any], l: dict[str, Any]) -> str:
    wf = as_bool(w.get("fall_like"))
    lf = as_bool(l.get("fall_like"))

    if not wf and lf:
        return "slide_reward_winner_did_not_fall_loser_fell"
    if not wf and not lf:
        return "slide_reward_both_stable"
    if wf and lf:
        return "slide_reward_both_fell_lower_damage"
    return "slide_reward_higher_score"


def pair_type(w: dict[str, Any], l: dict[str, Any]) -> str:
    wf = str(w.get("primitive_family", "unknown"))
    lf = str(l.get("primitive_family", "unknown"))
    return f"{wf}__over__{lf}"


def confidence_from_gap(
    gap: float,
    w: dict[str, Any],
    l: dict[str, Any],
    scale: float,
) -> float:
    wf = as_bool(w.get("fall_like"))
    lf = as_bool(l.get("fall_like"))

    if not wf and lf:
        return 1.0

    return max(0.05, min(1.0, gap / max(scale, 1e-9)))


def build_pairs(
    rows: list[dict[str, Any]],
    *,
    min_reward_gap: float,
    confidence_scale: float,
    drop_both_fell: bool,
    max_pairs: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda r: as_float(r.get("R_tracer_slide_v0"), -1.0),
        reverse=True,
    )

    pairs: list[dict[str, Any]] = []

    for i, winner in enumerate(ranked):
        rw = as_float(winner.get("R_tracer_slide_v0"), -1.0)

        for j in range(i + 1, len(ranked)):
            loser = ranked[j]
            rl = as_float(loser.get("R_tracer_slide_v0"), -1.0)
            gap = rw - rl

            if gap < min_reward_gap:
                continue

            if drop_both_fell and as_bool(winner.get("fall_like")) and as_bool(loser.get("fall_like")):
                continue

            pair = {
                "pair_id": f"slide_pref_{len(pairs):06d}",
                "reason": pair_reason(winner, loser),
                "pair_type": pair_type(winner, loser),
                "confidence": confidence_from_gap(gap, winner, loser, confidence_scale),
                "winner_reward": rw,
                "loser_reward": rl,
                "reward_gap": gap,
                "winner_tag": winner.get("tag"),
                "winner_style": winner.get("style"),
                "winner_primitive_family": winner.get("primitive_family"),
                "winner_fall_like": winner.get("fall_like"),
                "loser_tag": loser.get("tag"),
                "loser_style": loser.get("style"),
                "loser_primitive_family": loser.get("primitive_family"),
                "loser_fall_like": loser.get("fall_like"),
            }
            pair.update(prefixed("winner", winner))
            pair.update(prefixed("loser", loser))
            pairs.append(pair)

            if max_pairs > 0 and len(pairs) >= max_pairs:
                return pairs

    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reward-csv",
        default="data/rollout_rewards/tracer_slide_reward_v0_filtered.csv",
    )
    ap.add_argument(
        "--out-dir",
        default="data/preference_datasets",
    )
    ap.add_argument("--min-reward-gap", type=float, default=0.01)
    ap.add_argument("--confidence-scale", type=float, default=0.10)
    ap.add_argument("--drop-both-fell", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    rows = read_csv(Path(args.reward_csv))
    pairs = build_pairs(
        rows,
        min_reward_gap=args.min_reward_gap,
        confidence_scale=args.confidence_scale,
        drop_both_fell=args.drop_both_fell,
        max_pairs=args.max_pairs,
    )

    suffix = "clean" if args.drop_both_fell else "all"

    out_dir = Path(args.out_dir)
    out_jsonl = out_dir / f"tracer_slide_reward_preference_pairs_v0_{suffix}.jsonl"
    out_csv = out_dir / f"tracer_slide_reward_preference_pairs_v0_{suffix}.csv"

    write_jsonl(out_jsonl, pairs)
    write_csv(out_csv, pairs)

    print(f"[TRACER] reward rows: {len(rows)}")
    print(f"[TRACER] pairs:       {len(pairs)}")
    print(f"[TRACER] wrote jsonl: {out_jsonl}")
    print(f"[TRACER] wrote csv:   {out_csv}")

    print("\n[TRACER] preview:")
    for p in pairs[:20]:
        print(
            f"{p['pair_id']} "
            f"{p['pair_type']} "
            f"conf={float(p['confidence']):.2f} "
            f"gap={float(p['reward_gap']):.4f} "
            f"winner={p['winner_style']} "
            f"loser={p['loser_style']} "
            f"reason={p['reason']}"
        )


if __name__ == "__main__":
    main()
