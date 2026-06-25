#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fields = [
        "tag",
        "npz_path",
        "summary_path",
        "world",
        "terrain_key",
        "policy_json",
        "style",
        "semantic_mode",
        "primitive_family",
        "command_vx",
        "command_yaw",
        "command_body_height",
        "command_clearance",
        "command_enable",
        "fall_like",
        "trajectory_cost_v0",
        "dx",
        "dy",
        "min_z",
        "max_abs_roll",
        "max_abs_pitch",
        "pose_source",
        "commit_hash",
        "source",
        "note",
    ]

    extra = sorted({k for r in rows for k in r.keys()} - set(fields))
    fields = fields + extra

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def infer_primitive_family(row: dict[str, Any]) -> str:
    vx = float(row.get("mpc_vx_mean") or 0.0)
    h = float(row.get("mpc_body_height_mean") or 0.0)
    clr = float(row.get("mpc_clearance_mean") or 0.0)
    enable = float(row.get("mpc_enable_mean") or 0.0)

    if enable < 0.5:
        return "passive_or_disabled_hold"
    if vx < -1e-4:
        return "negative_vx_backstep"
    if abs(vx) <= 1e-4 and h > 0.0:
        return "active_hold"
    if vx > 1e-4:
        return "forward_locomotion"
    return "unknown"


def command_signature(row: dict[str, Any]) -> str:
    vx = float(row.get("mpc_vx_mean") or 0.0)
    h = float(row.get("mpc_body_height_mean") or 0.0)
    clr = float(row.get("mpc_clearance_mean") or 0.0)
    enable = float(row.get("mpc_enable_mean") or 0.0)
    return f"vx={vx:.4f},h={h:.3f},clr={clr:.3f},enable={enable:.1f}"


def default_note(row: dict[str, Any]) -> str:
    family = infer_primitive_family(row)
    sig = command_signature(row)
    return f"auto_manifest_from_metrics; primitive_family={family}; {sig}"


def merge_override(row: dict[str, Any], overrides: dict[str, dict[str, str]]) -> dict[str, Any]:
    tag = str(row.get("tag", ""))
    out = dict(row)

    # Defaults from metrics.
    out.setdefault("world", "unknown")
    out.setdefault("terrain_key", "unknown")
    out.setdefault("policy_json", "unknown")
    out.setdefault("style", "unknown")
    out.setdefault("semantic_mode", "unknown")
    out.setdefault("primitive_family", infer_primitive_family(row))
    out.setdefault("command_vx", row.get("mpc_vx_mean", ""))
    out.setdefault("command_yaw", row.get("mpc_yaw_mean", ""))
    out.setdefault("command_body_height", row.get("mpc_body_height_mean", ""))
    out.setdefault("command_clearance", row.get("mpc_clearance_mean", ""))
    out.setdefault("command_enable", row.get("mpc_enable_mean", ""))
    out.setdefault("commit_hash", git_commit_hash())
    out.setdefault("source", "auto")
    out.setdefault("note", default_note(row))

    # Manual override by tag.
    if tag in overrides:
        for k, v in overrides[tag].items():
            if v != "":
                out[k] = v
        out["source"] = "manual_override"

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", default="data/rollout_metrics/tracer_rollout_metrics_v0.csv")
    ap.add_argument("--manual-csv", default="configs/rollout_metadata/tracer_rollout_manual_manifest_v0.csv")
    ap.add_argument("--out-dir", default="data/rollout_manifests")
    args = ap.parse_args()

    metrics_csv = Path(args.metrics_csv)
    manual_csv = Path(args.manual_csv)
    out_dir = Path(args.out_dir)

    metrics = read_csv(metrics_csv)
    manual = read_csv(manual_csv)
    overrides = {str(r.get("tag", "")): r for r in manual if r.get("tag")}

    rows = [merge_override(r, overrides) for r in metrics]

    out_csv = out_dir / "tracer_rollout_manifest_v0.csv"
    out_jsonl = out_dir / "tracer_rollout_manifest_v0.jsonl"

    write_csv(out_csv, rows)
    write_jsonl(out_jsonl, rows)

    print(f"[TRACER] metrics rows:   {len(metrics)}")
    print(f"[TRACER] manual rows:    {len(manual)}")
    print(f"[TRACER] manifest rows:  {len(rows)}")
    print(f"[TRACER] wrote csv:      {out_csv}")
    print(f"[TRACER] wrote jsonl:    {out_jsonl}")

    print("\n[TRACER] manifest preview:")
    for r in rows[:20]:
        print(
            f"{r.get('tag')} "
            f"family={r.get('primitive_family')} "
            f"world={r.get('world')} "
            f"terrain={r.get('terrain_key')} "
            f"style={r.get('style')} "
            f"cost={r.get('trajectory_cost_v0')} "
            f"fall={r.get('fall_like')}"
        )


if __name__ == "__main__":
    main()
