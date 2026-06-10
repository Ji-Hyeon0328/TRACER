#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import csv
import json
import glob
import math

try:
    import numpy as np
except Exception:
    np = None


MODE_TO_ID = {
    "nominal": 0,
    "cautious_mismatch": 1,
    "conservative_mismatch": 2,
    "conservative_posture": 3,
    "recovery": 4,
    "reached": 5,
    "": -1,
}


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def load_summary_valid_map(log_dir):
    summary_path = os.path.join(log_dir, "tracer_goal_tracer_summary.csv")
    valid = {}

    if not os.path.exists(summary_path):
        return valid

    with open(summary_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            fname = row.get("file", "")
            valid[fname] = safe_int(row.get("valid_for_report", 0))

    return valid


def read_log(path):
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            phase = row.get("phase", "")
            if phase not in ("track", "reached", ""):
                continue
            rows.append(row)
    return rows


def get_goal_dist(row):
    gx = safe_float(row.get("goal_x", 0.0))
    gy = safe_float(row.get("goal_y", 0.0))
    x = safe_float(row.get("x", 0.0))
    y = safe_float(row.get("y", 0.0))
    return math.sqrt((gx - x) ** 2 + (gy - y) ** 2)


def get_feature(row, field):
    if field == "goal_dist":
        return get_goal_dist(row)

    if field == "vx_mod_error":
        raw = safe_float(row.get("raw_vx_axis", 0.0))
        mod = safe_float(row.get("vx_axis", 0.0))
        return raw - mod

    if field == "yaw_mod_error":
        raw = safe_float(row.get("raw_yaw_axis", 0.0))
        mod = safe_float(row.get("yaw_axis", 0.0))
        return raw - mod

    if field == "mode_id":
        return float(MODE_TO_ID.get(row.get("mode", ""), -1))

    return safe_float(row.get(field, 0.0))


def build_samples_from_rows(rows, feature_fields, label_fields, history_len, stride):
    X = []
    Y = []
    meta = []

    if len(rows) < history_len:
        return X, Y, meta

    for end_idx in range(history_len - 1, len(rows), stride):
        start_idx = end_idx - history_len + 1
        hist = rows[start_idx:end_idx + 1]
        target_row = rows[end_idx]

        x_seq = []
        bad = False

        for row in hist:
            feats = []
            for field in feature_fields:
                val = get_feature(row, field)
                if math.isnan(val) or math.isinf(val):
                    bad = True
                    break
                feats.append(val)
            if bad:
                break
            x_seq.append(feats)

        if bad:
            continue

        y = []
        for field in label_fields:
            val = safe_float(target_row.get(field, 0.0))
            if math.isnan(val) or math.isinf(val):
                bad = True
                break
            y.append(val)

        if bad:
            continue

        X.append(x_seq)
        Y.append(y)

        meta.append({
            "ros_time": target_row.get("ros_time", ""),
            "mode": target_row.get("mode", ""),
            "goal_dist": get_goal_dist(target_row),
            "file_row_idx": end_idx,
        })

    return X, Y, meta


def write_meta_csv(path, meta, source_files):
    with open(path, "w") as f:
        fields = ["sample_idx", "source_file", "file_row_idx", "ros_time", "mode", "goal_dist"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for i, m in enumerate(meta):
            row = {
                "sample_idx": i,
                "source_file": source_files[i],
                "file_row_idx": m.get("file_row_idx", ""),
                "ros_time": m.get("ros_time", ""),
                "mode": m.get("mode", ""),
                "goal_dist": "%.6f" % safe_float(m.get("goal_dist", 0.0)),
            }
            w.writerow(row)


def main():
    if len(sys.argv) < 2:
        print("Usage: python tracer_core/tools/build_ram_dataset_from_goal_logs.py <config.json>")
        sys.exit(1)

    cfg_path = sys.argv[1]

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    log_dir = cfg.get("log_dir", "/root/TRACER/logs")
    output_dir = cfg.get("output_dir", "/root/TRACER/datasets/ram")
    output_name = cfg.get("output_name", "ram_goal_tracer_v0")

    only_valid = bool(cfg.get("only_valid_for_report", True))
    only_cfg = bool(cfg.get("only_cfg_logs", False))

    history_len = safe_int(cfg.get("history_len", 40), 40)
    stride = safe_int(cfg.get("stride", 1), 1)

    feature_fields = cfg.get("feature_fields", [])
    label_fields = cfg.get("label_fields", [])

    if not feature_fields:
        print("No feature_fields provided.")
        sys.exit(1)

    if not label_fields:
        print("No label_fields provided.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_map = load_summary_valid_map(log_dir)

    paths = sorted(glob.glob(os.path.join(log_dir, "tracer_goal_tracer_*.csv")))

    all_X = []
    all_Y = []
    all_meta = []
    all_source_files = []

    used_files = []
    skipped_files = []

    for path in paths:
        fname = os.path.basename(path)

        # Skip summary/report files.
        if "summary" in fname or "report" in fname:
            continue

        if only_cfg and "cfg_" not in fname:
            skipped_files.append((fname, "not cfg log"))
            continue

        if only_valid:
            if valid_map.get(fname, 0) != 1:
                skipped_files.append((fname, "not valid_for_report"))
                continue

        rows = read_log(path)

        if len(rows) < history_len:
            skipped_files.append((fname, "too short"))
            continue

        X, Y, meta = build_samples_from_rows(
            rows,
            feature_fields,
            label_fields,
            history_len,
            stride
        )

        if not X:
            skipped_files.append((fname, "no samples"))
            continue

        all_X.extend(X)
        all_Y.extend(Y)
        all_meta.extend(meta)
        all_source_files.extend([fname] * len(X))
        used_files.append((fname, len(X)))

    if not all_X:
        print("No samples collected.")
        print("Skipped files:")
        for item in skipped_files:
            print("  %s: %s" % item)
        sys.exit(0)

    if np is None:
        print("numpy is not available. Cannot write npz.")
        sys.exit(1)

    X_arr = np.asarray(all_X, dtype=np.float32)
    Y_arr = np.asarray(all_Y, dtype=np.float32)

    npz_path = os.path.join(output_dir, output_name + ".npz")
    meta_path = os.path.join(output_dir, output_name + "_meta.csv")
    info_path = os.path.join(output_dir, output_name + "_info.json")

    np.savez(
        npz_path,
        X=X_arr,
        Y=Y_arr,
        feature_fields=np.asarray(feature_fields),
        label_fields=np.asarray(label_fields)
    )

    write_meta_csv(meta_path, all_meta, all_source_files)

    info = {
        "config": cfg,
        "num_samples": int(X_arr.shape[0]),
        "history_len": int(X_arr.shape[1]),
        "feature_dim": int(X_arr.shape[2]),
        "label_dim": int(Y_arr.shape[1]),
        "feature_fields": feature_fields,
        "label_fields": label_fields,
        "used_files": used_files,
        "skipped_files": skipped_files,
        "npz_path": npz_path,
        "meta_path": meta_path
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, sort_keys=True)

    print("Wrote:", npz_path)
    print("Wrote:", meta_path)
    print("Wrote:", info_path)
    print("")
    print("X shape:", X_arr.shape)
    print("Y shape:", Y_arr.shape)
    print("features:", feature_fields)
    print("labels:", label_fields)
    print("")
    print("Used files:")
    for fname, n in used_files:
        print("  %s: %d samples" % (fname, n))

    print("")
    print("Skipped files:")
    for fname, reason in skipped_files:
        print("  %s: %s" % (fname, reason))


if __name__ == "__main__":
    main()
