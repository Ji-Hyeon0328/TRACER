#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def pick(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def as_float(x, default=float("nan")):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data/style_missions",
        help="Directory containing <run_id>_<style>/ folders",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: <root>/style_mission_summary.csv",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out) if args.out else root / "style_mission_summary.csv"
    out_md = out_csv.with_suffix(".md")

    rows = []

    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir():
            continue

        meta = load_json(run_dir / "style_mission_metadata.json")
        summ = load_json(run_dir / "mission_summary.json")

        style_cmd = meta.get("style_command", {})

        # mission_summary keys may evolve, so use tolerant key aliases.
        success = pick(summ, ["success"], None)
        if success is None:
            success = meta.get("mission_status") == "success"

        row = {
            "run_dir": str(run_dir),
            "run_id": meta.get("run_id", run_dir.name),
            "style": meta.get("style_name", run_dir.name.split("_")[-1]),
            "status": meta.get("mission_status", "unknown"),
            "success": bool(success),
            "duration": as_float(pick(summ, ["duration", "duration_sec", "elapsed_sec"])),
            "path": as_float(pick(summ, ["path", "path_length", "path_length_m"])),
            "net_disp": as_float(pick(summ, ["net_disp", "net_displacement", "net_displacement_m"])),
            "vx_cmd": as_float(style_cmd.get("vx")),
            "yaw_cmd": as_float(style_cmd.get("yaw_rate")),
            "body_height_cmd": as_float(style_cmd.get("body_height")),
            "clearance_cmd": as_float(style_cmd.get("swing_clearance")),
            "beta_hint": ",".join(str(x) for x in meta.get("beta_hint", [])),
        }

        # Derived path efficiency.
        if math.isfinite(row["path"]) and math.isfinite(row["net_disp"]) and row["path"] > 1e-6:
            row["path_eff"] = row["net_disp"] / row["path"]
        else:
            row["path_eff"] = float("nan")

        rows.append(row)

    fields = [
        "run_id",
        "style",
        "status",
        "success",
        "duration",
        "path",
        "net_disp",
        "path_eff",
        "vx_cmd",
        "body_height_cmd",
        "clearance_cmd",
        "beta_hint",
        "run_dir",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    def fmt(x):
        if isinstance(x, float):
            if math.isnan(x):
                return ""
            return f"{x:.3f}"
        return str(x)

    lines = []
    lines.append("# TRACER Style Mission Summary")
    lines.append("")
    lines.append("| run_id | style | status | success | duration | path | net disp | path eff | vx | height | clearance |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {run_id} | {style} | {status} | {success} | {duration} | {path} | {net_disp} | {path_eff} | {vx_cmd} | {body_height_cmd} | {clearance_cmd} |".format(
                run_id=r["run_id"],
                style=r["style"],
                status=r["status"],
                success=r["success"],
                duration=fmt(r["duration"]),
                path=fmt(r["path"]),
                net_disp=fmt(r["net_disp"]),
                path_eff=fmt(r["path_eff"]),
                vx_cmd=fmt(r["vx_cmd"]),
                body_height_cmd=fmt(r["body_height_cmd"]),
                clearance_cmd=fmt(r["clearance_cmd"]),
            )
        )

    out_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] found {len(rows)} style mission runs")
    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_md}")
    print()
    print(out_md.read_text())


if __name__ == "__main__":
    main()
