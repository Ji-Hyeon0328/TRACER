#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def choose_item(items: list[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    if not items:
        return None

    mode = mode.lower()

    if mode in {"fast", "preferred_fast"}:
        fast = [x for x in items if x.get("group_label") == "preferred_fast"]
        if fast:
            return sorted(fast, key=lambda x: f(x.get("mean_cost", 9999.0)))[0]
        return sorted(items, key=lambda x: f(x.get("mean_cost", 9999.0)))[0]

    if mode in {"balanced", "preferred_balanced"}:
        bal = [x for x in items if x.get("group_label") == "preferred_balanced"]
        if bal:
            # For balanced, prefer low drift / stable attitude, then cost.
            return sorted(
                bal,
                key=lambda x: (
                    f(x.get("mean_abs_dy_per_dx", 9999.0)),
                    f(x.get("mean_pitch_abs_max", 9999.0)),
                    f(x.get("mean_cost", 9999.0)),
                ),
            )[0]
        return sorted(items, key=lambda x: f(x.get("mean_cost", 9999.0)))[0]

    if mode in {"conservative", "safe", "cautious"}:
        # Conservative: prefer higher body height, low drift, lower vx.
        candidates = sorted(
            items,
            key=lambda x: (
                -f(x.get("body_height", 0.0)),
                f(x.get("mean_abs_dy_per_dx", 9999.0)),
                f(x.get("vx", 9999.0)),
                f(x.get("mean_cost", 9999.0)),
            ),
        )
        return candidates[0]

    # Default: lowest mean cost.
    return sorted(items, key=lambda x: f(x.get("mean_cost", 9999.0)))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/rollout_metrics/gms_reference_table_grouped_v1.json")
    ap.add_argument("--terrain", default="flat_normal")
    ap.add_argument("--mode", default="balanced", choices=["fast", "balanced", "conservative", "safe", "cautious"])
    ap.add_argument("--bucket", default="preferred", choices=["preferred", "acceptable"])
    ap.add_argument("--yaw-rate", type=float, default=0.0)
    ap.add_argument("--counter", type=float, default=0.0)
    ap.add_argument("--enable", type=float, default=1.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    table = json.loads(Path(args.table).read_text())
    terrains = table.get("terrains", {})
    if args.terrain not in terrains:
        raise SystemExit(f"[ERROR] terrain not found in table: {args.terrain}. available={list(terrains.keys())}")

    terrain_entry = terrains[args.terrain]
    items = list(terrain_entry.get(args.bucket, []))

    # If preferred bucket is empty, fall back to acceptable.
    if not items and args.bucket == "preferred":
        items = list(terrain_entry.get("acceptable", []))

    # Fast mode has a special fallback:
    # if no preferred_fast exists in preferred bucket, search acceptable candidates
    # and choose the highest-vx acceptable reference with reasonable cost.
    if args.mode == "fast":
        preferred_fast = [x for x in items if x.get("group_label") == "preferred_fast"]
        if not preferred_fast:
            acc = list(terrain_entry.get("acceptable", []))
            if acc:
                item = sorted(
                    acc,
                    key=lambda x: (
                        -f(x.get("vx", 0.0)),
                        f(x.get("mean_cost", 9999.0)),
                        f(x.get("mean_abs_dy_per_dx", 9999.0)),
                    ),
                )[0]
            else:
                item = choose_item(items, args.mode)
        else:
            item = choose_item(items, args.mode)
    else:
        item = choose_item(items, args.mode)
    if item is None:
        raise SystemExit(f"[ERROR] no selectable item for terrain={args.terrain}, bucket={args.bucket}")

    ref = [
        float(args.counter),
        f(item.get("vx")),
        float(args.yaw_rate),
        f(item.get("body_height")),
        f(item.get("clearance")),
        float(args.enable),
    ]

    out = {
        "terrain": args.terrain,
        "mode": args.mode,
        "bucket": args.bucket,
        "selected": item,
        "mpc_reference": ref,
        "layout": ["counter", "vx", "yaw_rate", "body_height", "swing_clearance", "enable"],
    }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print("selected:", json.dumps(item, indent=2))
        print("mpc_reference:", ref)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
