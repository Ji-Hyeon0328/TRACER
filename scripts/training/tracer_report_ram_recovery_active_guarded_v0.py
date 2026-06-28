#!/usr/bin/env python3
"""Report RAM recovery active candidate v0 guarded rollout results.

This utility summarizes rollout episode summary JSON files and checks whether
the guarded active recovery path behaved as expected.

Default expectation:
  - active recovery may be enabled and apply-capable,
  - but in normal flat/rough/slope robust runs it should remain no-op:
      shadow trigger == 0
      active trigger == 0
      active applied == 0
  - success_proxy should pass for all selected episodes.

This is a report utility only. It does not modify rollout data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def b(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return False


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def max0(xs: list[float]) -> float:
    return max(xs) if xs else 0.0


def min0(xs: list[float]) -> float:
    return min(xs) if xs else 0.0


def load_summary(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text())
        if not isinstance(obj, dict):
            return None
        obj["_path"] = str(path)
        obj["_mtime"] = path.stat().st_mtime
        return obj
    except Exception as exc:
        print(f"[WARN] failed to read {path}: {exc!r}")
        return None


def split_tokens(s: str) -> list[str]:
    return [x.strip() for x in s.replace(",", " ").split() if x.strip()]


def select_latest_per_terrain(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0:
        return rows
    by_terrain: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        terrain = str(r.get("terrain", "unknown"))
        by_terrain.setdefault(terrain, []).append(r)

    out: list[dict[str, Any]] = []
    for terrain, items in sorted(by_terrain.items()):
        items = sorted(items, key=lambda x: f(x.get("_mtime", 0.0)), reverse=True)
        out.extend(items[:n])
    return sorted(out, key=lambda x: (str(x.get("terrain", "")), str(x.get("episode_id", ""))))


def summarize_group(
    terrain: str,
    rows: list[dict[str, Any]],
    *,
    min_success_rate: float,
    min_debug_fresh_rate: float,
    expect_no_active: bool,
) -> dict[str, Any]:
    n = len(rows)
    success = [1.0 if b(r.get("success_proxy", False)) else 0.0 for r in rows]

    shadow_trigger_max = max0([f(r.get("shadow_recovery_trigger_count_max")) for r in rows])
    shadow_active_rows_max = max0([f(r.get("shadow_recovery_active_rows_max")) for r in rows])
    shadow_would_mean_avg = mean([f(r.get("shadow_recovery_would_recover_mean")) for r in rows])

    active_enabled_avg = mean([f(r.get("active_recovery_enabled_mean")) for r in rows])
    active_apply_avg = mean([f(r.get("active_recovery_apply_mean")) for r in rows])
    active_active_avg = mean([f(r.get("active_recovery_active_mean")) for r in rows])
    active_applied_avg = mean([f(r.get("active_recovery_applied_mean")) for r in rows])
    active_trigger_max = max0([f(r.get("active_recovery_trigger_count_max")) for r in rows])
    active_rows_max = max0([f(r.get("active_recovery_active_rows_max")) for r in rows])
    active_protected_avg = mean([f(r.get("active_recovery_protected_mean")) for r in rows])

    debug_fresh_min = min0([f(r.get("debug_fresh_rate_1p0s")) for r in rows])
    debug_p90_avg = mean([f(r.get("age_debug_p90")) for r in rows])

    vx_avg = mean([f(r.get("mpc_vx_mean")) for r in rows])
    enable_avg = mean([f(r.get("mpc_enable_mean")) for r in rows])
    recovery_score_p90_max = max0([f(r.get("active_recovery_score_p90")) for r in rows])

    success_rate = mean(success)

    fail_reasons: list[str] = []
    if n <= 0:
        fail_reasons.append("no_episodes")
    if success_rate + 1e-12 < min_success_rate:
        fail_reasons.append(f"success_rate<{min_success_rate:g}")
    if debug_fresh_min + 1e-12 < min_debug_fresh_rate:
        fail_reasons.append(f"debug_fresh_1p0s<{min_debug_fresh_rate:g}")
    if expect_no_active:
        if shadow_trigger_max > 0.0:
            fail_reasons.append("shadow_trigger_nonzero")
        if shadow_active_rows_max > 0.0:
            fail_reasons.append("shadow_active_rows_nonzero")
        if active_trigger_max > 0.0:
            fail_reasons.append("active_trigger_nonzero")
        if active_rows_max > 0.0:
            fail_reasons.append("active_rows_nonzero")
        if active_applied_avg > 1e-12:
            fail_reasons.append("active_applied_nonzero")

    return {
        "terrain": terrain,
        "episodes": n,
        "success_count": int(sum(success)),
        "success_rate": success_rate,
        "mpc_vx_mean_avg": vx_avg,
        "mpc_enable_mean_avg": enable_avg,
        "shadow_would_recover_mean_avg": shadow_would_mean_avg,
        "shadow_trigger_count_max": shadow_trigger_max,
        "shadow_active_rows_max": shadow_active_rows_max,
        "active_enabled_mean_avg": active_enabled_avg,
        "active_apply_mean_avg": active_apply_avg,
        "active_active_mean_avg": active_active_avg,
        "active_applied_mean_avg": active_applied_avg,
        "active_trigger_count_max": active_trigger_max,
        "active_active_rows_max": active_rows_max,
        "active_protected_mean_avg": active_protected_avg,
        "active_score_p90_max": recovery_score_p90_max,
        "debug_fresh_rate_1p0s_min": debug_fresh_min,
        "age_debug_p90_avg": debug_p90_avg,
        "pass": len(fail_reasons) == 0,
        "fail_reasons": ",".join(fail_reasons),
        "episode_ids": [str(r.get("episode_id", "")) for r in rows],
        "summary_paths": [str(r.get("_path", "")) for r in rows],
    }


def print_table(groups: list[dict[str, Any]]) -> None:
    cols = [
        ("terrain", 16),
        ("episodes", 8),
        ("success", 9),
        ("vx", 8),
        ("shadow_trig", 12),
        ("active_trig", 12),
        ("active_apply", 12),
        ("fresh1s", 8),
        ("PASS", 6),
    ]

    header = " ".join(name.ljust(width) for name, width in cols)
    print(header)
    print("-" * len(header))

    for g in groups:
        row = {
            "terrain": str(g["terrain"]),
            "episodes": str(g["episodes"]),
            "success": f"{g['success_count']}/{g['episodes']}",
            "vx": f"{g['mpc_vx_mean_avg']:.3f}",
            "shadow_trig": f"{g['shadow_trigger_count_max']:.0f}",
            "active_trig": f"{g['active_trigger_count_max']:.0f}",
            "active_apply": f"{g['active_applied_mean_avg']:.3f}",
            "fresh1s": f"{g['debug_fresh_rate_1p0s_min']:.3f}",
            "PASS": "PASS" if g["pass"] else "FAIL",
        }
        print(" ".join(row[name].ljust(width)[:width] for name, width in cols))
        if not g["pass"]:
            print(f"  fail_reasons: {g['fail_reasons']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", default="data/rollout_dataset_v0/summaries")
    ap.add_argument("--glob", default="*learned_stack_v3_beta_blend_robust30*.json")
    ap.add_argument("--contains", action="append", default=[], help="Require substring in filename, can be repeated.")
    ap.add_argument("--terrains", default="", help="Space/comma separated terrain names to include.")
    ap.add_argument("--latest-per-terrain", type=int, default=3)
    ap.add_argument("--min-success-rate", type=float, default=1.0)
    ap.add_argument("--min-debug-fresh-rate", type=float, default=0.90)
    ap.add_argument("--allow-active", action="store_true", help="Do not fail if active recovery triggers/applies.")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--csv-out", default="")
    ap.add_argument("--no-strict-exit", action="store_true")
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    paths = sorted(summary_dir.glob(args.glob))
    if args.contains:
        paths = [p for p in paths if all(c in p.name for c in args.contains)]

    rows = [r for p in paths if (r := load_summary(p)) is not None]

    terrain_filter = set(split_tokens(args.terrains))
    if terrain_filter:
        rows = [r for r in rows if str(r.get("terrain", "")) in terrain_filter]

    rows = select_latest_per_terrain(rows, args.latest_per_terrain)

    by_terrain: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_terrain.setdefault(str(r.get("terrain", "unknown")), []).append(r)

    groups = [
        summarize_group(
            terrain,
            items,
            min_success_rate=args.min_success_rate,
            min_debug_fresh_rate=args.min_debug_fresh_rate,
            expect_no_active=not args.allow_active,
        )
        for terrain, items in sorted(by_terrain.items())
    ]

    print("[TRACER] RAM recovery active guarded report")
    print(f"[TRACER] summary_dir={summary_dir}")
    print(f"[TRACER] glob={args.glob!r}")
    print(f"[TRACER] selected_episodes={len(rows)} terrains={len(groups)}")
    print(f"[TRACER] latest_per_terrain={args.latest_per_terrain}")
    print(f"[TRACER] expect_no_active={int(not args.allow_active)}")
    print()
    print_table(groups)

    overall_pass = bool(groups) and all(g["pass"] for g in groups)
    print()
    print(f"[TRACER] OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    payload = {
        "overall_pass": overall_pass,
        "selected_episodes": len(rows),
        "groups": groups,
    }

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"[TRACER] wrote json: {out}")

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = [
            "terrain",
            "episodes",
            "success_count",
            "success_rate",
            "mpc_vx_mean_avg",
            "mpc_enable_mean_avg",
            "shadow_would_recover_mean_avg",
            "shadow_trigger_count_max",
            "shadow_active_rows_max",
            "active_enabled_mean_avg",
            "active_apply_mean_avg",
            "active_active_mean_avg",
            "active_applied_mean_avg",
            "active_trigger_count_max",
            "active_active_rows_max",
            "active_protected_mean_avg",
            "active_score_p90_max",
            "debug_fresh_rate_1p0s_min",
            "age_debug_p90_avg",
            "pass",
            "fail_reasons",
        ]
        with out.open("w", newline="") as fobj:
            w = csv.DictWriter(fobj, fieldnames=keys)
            w.writeheader()
            for g in groups:
                w.writerow({k: g.get(k, "") for k in keys})
        print(f"[TRACER] wrote csv: {out}")

    if overall_pass or args.no_strict_exit:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
