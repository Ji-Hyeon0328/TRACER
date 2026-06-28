#!/usr/bin/env python3
"""Report RAM head calibration patterns from rollout episode summaries.

This utility is meant to catch cases where RAM heads disagree with rollout
outcomes, for example:
  - success_proxy is true but ram_run_fallen is high
  - recovery_needed is low while run_fallen is high
  - shadow/active recovery never triggers despite a high fall head

It is diagnostic only and does not modify data.
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


def split_tokens(s: str) -> list[str]:
    return [x.strip() for x in s.replace(",", " ").split() if x.strip()]


def load(path: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(path.read_text())
        if not isinstance(d, dict):
            return None
        d["_path"] = str(path)
        d["_mtime"] = path.stat().st_mtime
        return d
    except Exception as exc:
        print(f"[WARN] failed to read {path}: {exc!r}")
        return None


def select_latest_per_terrain(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0:
        return rows
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r.get("terrain", "unknown")), []).append(r)

    out: list[dict[str, Any]] = []
    for terrain, items in sorted(by.items()):
        items = sorted(items, key=lambda r: f(r.get("_mtime", 0.0)), reverse=True)
        out.extend(items[:n])
    return sorted(out, key=lambda r: (str(r.get("terrain", "")), str(r.get("episode_id", ""))))


def summarize(terrain: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    n = len(rows)
    success = [1.0 if b(r.get("success_proxy", False)) else 0.0 for r in rows]

    run_fallen_mean = [f(r.get("ram_run_fallen_mean")) for r in rows]
    run_fallen_p90 = [f(r.get("ram_run_fallen_p90")) for r in rows]
    recovery_mean = [f(r.get("ram_recovery_needed_mean")) for r in rows]
    shadow_score = [f(r.get("shadow_recovery_score_mean")) for r in rows]
    active_score = [f(r.get("active_recovery_score_mean")) for r in rows]

    shadow_trigger = [f(r.get("shadow_recovery_trigger_count_max")) for r in rows]
    active_trigger = [f(r.get("active_recovery_trigger_count_max")) for r in rows]
    active_applied = [f(r.get("active_recovery_applied_mean")) for r in rows]

    debug_fresh = [f(r.get("debug_fresh_rate_1p0s")) for r in rows]
    vx = [f(r.get("mpc_vx_mean")) for r in rows]
    enable = [f(r.get("mpc_enable_mean")) for r in rows]
    proprio_abs = [f(r.get("proprio_abs_mean")) for r in rows]

    success_rate = mean(success)
    run_fallen_mean_avg = mean(run_fallen_mean)
    run_fallen_p90_max = max0(run_fallen_p90)
    recovery_mean_avg = mean(recovery_mean)

    flags: list[str] = []
    if success_rate >= args.success_high and run_fallen_mean_avg >= args.fallen_mean_high:
        flags.append("success_high_but_run_fallen_mean_high")
    if success_rate >= args.success_high and run_fallen_p90_max >= args.fallen_p90_high:
        flags.append("success_high_but_run_fallen_p90_high")
    if run_fallen_mean_avg >= args.fallen_mean_high and recovery_mean_avg <= args.recovery_low:
        flags.append("run_fallen_high_but_recovery_low")
    if max0(shadow_trigger) == 0.0 and recovery_mean_avg <= args.recovery_low and run_fallen_mean_avg >= args.fallen_mean_high:
        flags.append("fallen_head_not_used_by_recovery_gate")
    if min0(debug_fresh) < args.min_debug_fresh:
        flags.append("debug_fresh_low")

    severity = "OK"
    if flags:
        severity = "WARN"
    if any("success_high_but" in x for x in flags):
        severity = "CHECK"

    return {
        "terrain": terrain,
        "episodes": n,
        "success_count": int(sum(success)),
        "success_rate": success_rate,
        "mpc_vx_mean_avg": mean(vx),
        "mpc_enable_mean_avg": mean(enable),
        "ram_run_fallen_mean_avg": run_fallen_mean_avg,
        "ram_run_fallen_p90_max": run_fallen_p90_max,
        "ram_recovery_needed_mean_avg": recovery_mean_avg,
        "shadow_recovery_score_mean_avg": mean(shadow_score),
        "active_recovery_score_mean_avg": mean(active_score),
        "shadow_trigger_count_max": max0(shadow_trigger),
        "active_trigger_count_max": max0(active_trigger),
        "active_applied_mean_avg": mean(active_applied),
        "debug_fresh_rate_1p0s_min": min0(debug_fresh),
        "proprio_abs_mean_avg": mean(proprio_abs),
        "severity": severity,
        "flags": ",".join(flags),
        "episode_ids": [str(r.get("episode_id", "")) for r in rows],
        "summary_paths": [str(r.get("_path", "")) for r in rows],
    }


def print_table(groups: list[dict[str, Any]]) -> None:
    cols = [
        ("terrain", 16),
        ("episodes", 8),
        ("success", 9),
        ("vx", 7),
        ("fallen_m", 9),
        ("fallen_p90", 10),
        ("recover_m", 10),
        ("sh_trig", 8),
        ("act_trig", 8),
        ("fresh", 7),
        ("severity", 8),
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
            "fallen_m": f"{g['ram_run_fallen_mean_avg']:.3f}",
            "fallen_p90": f"{g['ram_run_fallen_p90_max']:.3f}",
            "recover_m": f"{g['ram_recovery_needed_mean_avg']:.6f}",
            "sh_trig": f"{g['shadow_trigger_count_max']:.0f}",
            "act_trig": f"{g['active_trigger_count_max']:.0f}",
            "fresh": f"{g['debug_fresh_rate_1p0s_min']:.3f}",
            "severity": str(g["severity"]),
        }
        print(" ".join(row[name].ljust(width)[:width] for name, width in cols))
        if g.get("flags"):
            print(f"  flags: {g['flags']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", default="data/rollout_dataset_v0/summaries")
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--contains", action="append", default=[])
    ap.add_argument("--terrains", default="")
    ap.add_argument("--latest-per-terrain", type=int, default=3)
    ap.add_argument("--success-high", type=float, default=0.99)
    ap.add_argument("--fallen-mean-high", type=float, default=0.30)
    ap.add_argument("--fallen-p90-high", type=float, default=0.80)
    ap.add_argument("--recovery-low", type=float, default=0.05)
    ap.add_argument("--min-debug-fresh", type=float, default=0.90)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--csv-out", default="")
    args = ap.parse_args()

    paths = sorted(Path(args.summary_dir).glob(args.glob))
    if args.contains:
        paths = [p for p in paths if all(c in p.name for c in args.contains)]

    rows = [r for p in paths if (r := load(p)) is not None]

    terrain_filter = set(split_tokens(args.terrains))
    if terrain_filter:
        rows = [r for r in rows if str(r.get("terrain", "")) in terrain_filter]

    rows = select_latest_per_terrain(rows, args.latest_per_terrain)

    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r.get("terrain", "unknown")), []).append(r)

    groups = [summarize(t, items, args) for t, items in sorted(by.items())]

    print("[TRACER] RAM head calibration diagnostic report")
    print(f"[TRACER] selected_episodes={len(rows)} terrains={len(groups)}")
    print(f"[TRACER] glob={args.glob!r} latest_per_terrain={args.latest_per_terrain}")
    print()
    print_table(groups)

    n_check = sum(1 for g in groups if g["severity"] == "CHECK")
    n_warn = sum(1 for g in groups if g["severity"] == "WARN")
    print()
    print(f"[TRACER] severity_count: CHECK={n_check} WARN={n_warn} OK={len(groups)-n_check-n_warn}")

    payload = {"groups": groups, "selected_episodes": len(rows)}
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
            "ram_run_fallen_mean_avg",
            "ram_run_fallen_p90_max",
            "ram_recovery_needed_mean_avg",
            "shadow_recovery_score_mean_avg",
            "active_recovery_score_mean_avg",
            "shadow_trigger_count_max",
            "active_trigger_count_max",
            "active_applied_mean_avg",
            "debug_fresh_rate_1p0s_min",
            "proprio_abs_mean_avg",
            "severity",
            "flags",
        ]
        with out.open("w", newline="") as fobj:
            w = csv.DictWriter(fobj, fieldnames=keys)
            w.writeheader()
            for g in groups:
                w.writerow({k: g.get(k, "") for k in keys})
        print(f"[TRACER] wrote csv: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
