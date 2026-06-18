#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


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


def compute_path_from_xy(xy):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[0] < 2 or xy.shape[1] < 2:
        return float("nan"), float("nan")
    xy = xy[np.isfinite(xy).all(axis=1)]
    if xy.shape[0] < 2:
        return float("nan"), float("nan")
    d = np.diff(xy[:, :2], axis=0)
    step = np.linalg.norm(d, axis=1)
    path = float(np.sum(step))
    net = float(np.linalg.norm(xy[-1, :2] - xy[0, :2]))
    return path, net


def extract_xy_from_npz(npz_path):
    """
    Try to recover robot xy trajectory from mission logger npz.
    Expected best case: a topic like robot_odom_flat with columns:
      [stamp, x, y, yaw, vx, vy]
    This function is intentionally tolerant because logger schemas may evolve.
    """
    if not npz_path.exists():
        return None, "missing_npz"

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return None, f"npz_load_error:{e}"

    keys = list(data.files)

    # Prefer explicit odom / robot odom arrays.
    priority = []
    for k in keys:
        lk = k.lower()
        score = 0
        if "robot_odom_flat" in lk:
            score += 100
        if "odom" in lk:
            score += 50
        if "base" in lk and ("pos" in lk or "pose" in lk):
            score += 20
        if "proprio" in lk:
            score += 10
        if score > 0:
            priority.append((score, k))

    priority.sort(reverse=True)

    candidate_keys = [k for _, k in priority] + [k for k in keys if k not in {x[1] for x in priority}]

    for k in candidate_keys:
        try:
            arr = np.asarray(data[k])
        except Exception:
            continue

        if arr.dtype == object:
            try:
                arr = np.stack(arr)
            except Exception:
                continue

        if arr.ndim != 2 or arr.shape[0] < 2:
            continue

        lk = k.lower()

        # Most likely schema: [stamp, x, y, yaw, vx, vy]
        if ("odom" in lk or "robot_odom_flat" in lk) and arr.shape[1] >= 3:
            xy = arr[:, 1:3]
            return xy, f"{k}:cols1_2"

        # Proprio vector schema used in our bridge:
        # [stamp, base_x, base_y, base_z, roll, pitch, yaw, ...]
        if "proprio" in lk and arr.shape[1] >= 4:
            xy = arr[:, 1:3]
            return xy, f"{k}:proprio_cols1_2"

        # Fallback for arrays that look like x,y at columns 0,1.
        if arr.shape[1] >= 2:
            xy = arr[:, 0:2]
            # Reject near-constant or obviously timestamp-like x column.
            if np.nanstd(xy[:, 0]) < 1e-9 and np.nanstd(xy[:, 1]) < 1e-9:
                continue
            if np.nanmedian(np.abs(np.diff(xy[:, 0]))) > 1000:
                continue
            return xy, f"{k}:fallback_cols0_1"

    return None, "no_xy_candidate_keys=" + ",".join(keys[:20])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/style_missions")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out) if args.out else root / "style_mission_summary.csv"
    out_md = out_csv.with_suffix(".md")

    rows = []

    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("INVALID_"):
            continue

        meta = load_json(run_dir / "style_mission_metadata.json")
        summ = load_json(run_dir / "mission_summary.json")
        style_cmd = meta.get("style_command", {})

        success = pick(summ, ["success"], None)
        if success is None:
            success = meta.get("mission_status") == "success"

        duration = as_float(pick(summ, ["duration", "duration_sec", "elapsed_sec"]))
        path = as_float(pick(summ, ["path", "path_length", "path_length_m"]))
        net = as_float(pick(summ, ["net_disp", "net_displacement", "net_displacement_m"]))

        npz_source = ""
        if not (math.isfinite(path) and math.isfinite(net)):
            npz_path = run_dir / "mission_log.npz"
            xy, source = extract_xy_from_npz(npz_path)
            npz_source = source
            if xy is not None:
                path2, net2 = compute_path_from_xy(xy)
                if math.isfinite(path2):
                    path = path2
                if math.isfinite(net2):
                    net = net2

                if not math.isfinite(duration):
                    # If xy came from odom-like array with stamp in col 0, duration can be recovered.
                    try:
                        arr = np.load(npz_path, allow_pickle=True)
                    except Exception:
                        arr = None

        row = {
            "run_dir": str(run_dir),
            "run_id": meta.get("run_id", run_dir.name),
            "style": meta.get("style_name", run_dir.name.split("_")[-1]),
            "terrain": meta.get("terrain_name", "unknown"),
            "world": meta.get("world_name", "unknown"),
            "status": meta.get("mission_status", "unknown"),
            "success": bool(success),
            "duration": duration,
            "path": path,
            "net_disp": net,
            "vx_cmd": as_float(style_cmd.get("vx")),
            "yaw_cmd": as_float(style_cmd.get("yaw_rate")),
            "body_height_cmd": as_float(style_cmd.get("body_height")),
            "clearance_cmd": as_float(style_cmd.get("swing_clearance")),
            "beta_hint": ",".join(str(x) for x in meta.get("beta_hint", [])),
            "npz_xy_source": npz_source,
        }

        if math.isfinite(row["path"]) and math.isfinite(row["net_disp"]) and row["path"] > 1e-6:
            row["path_eff"] = row["net_disp"] / row["path"]
        else:
            row["path_eff"] = float("nan")

        rows.append(row)

    fields = [
        "run_id",
        "style",
        "terrain",
        "world",
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
        "npz_xy_source",
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
    lines.append("| run_id | terrain | style | status | success | duration | path | net disp | path eff | vx | height | clearance |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {run_id} | {terrain} | {style} | {status} | {success} | {duration} | {path} | {net_disp} | {path_eff} | {vx_cmd} | {body_height_cmd} | {clearance_cmd} |".format(
                run_id=r["run_id"],
                terrain=r.get("terrain", "unknown"),
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
