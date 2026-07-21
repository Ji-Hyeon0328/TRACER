#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path
from statistics import mean, pstdev

IN_DIR = Path("datasets/phase_f")
OUT_CSV = Path("datasets/phase_f/f4_true_metric_training_table_v0.csv")
OUT_MD = Path("reports/phase_f/f4_true_metric_training_table_summary_v0.md")

def ff(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default

def vals(rows, *names):
    out = []
    for r in rows:
        v = None
        for n in names:
            if n in r:
                v = ff(r.get(n))
                if v is not None:
                    break
        if v is not None:
            out.append(v)
    return out

def safe_mean(x):
    return mean(x) if x else ""

def safe_std(x):
    return pstdev(x) if len(x) > 1 else 0.0 if x else ""

def safe_min(x):
    return min(x) if x else ""

def safe_max(x):
    return max(x) if x else ""

def infer_tag(path):
    m = re.match(r"f3_(.+?)_\d{8}_\d{6}_true_metrics\.csv", path.name)
    return m.group(1) if m else "unknown"

def base_tag(tag):
    return re.sub(r"_r[0-9]+$", "", tag)

def infer_reset_y(tag):
    b = base_tag(tag)
    if b == "clean":
        return 0.0
    m = re.match(r"m([0-9]+)$", b)
    if m:
        return -float(m.group(1)) / 100.0
    m = re.match(r"p([0-9]+)$", b)
    if m:
        return float(m.group(1)) / 100.0
    return ""

def latest_manifest_for_tag(tag):
    manifests = sorted(Path("reports").glob("phase_d4_context_meta_repeat_*_manifest.tsv"))
    for p in reversed(manifests):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        if tag in txt:
            return str(p)
    return ""

def find_summary_csv(true_csv):
    p = Path(str(true_csv).replace("_true_metrics.csv", "_true_metric_summary.csv"))
    return str(p) if p.exists() else ""

def parse_tag_stamp(path):
    m = re.match(r"f3_(.+?)_(\d{8}_\d{6})_true_metrics\.csv", path.name)
    if not m:
        return infer_tag(path), ""
    return m.group(1), m.group(2)

def rel_repo_path_string(x):
    x = str(x).strip()
    root = "/home/kraken/Tracer/TRACER/"
    if x.startswith(root):
        x = x.replace(root, "", 1)
    return x

def parse_run_log_metadata(tag, stamp):
    meta = {
        "run_log": "",
        "manifest": "",
        "d5_log_dir": "",
        "d7_csv": "",
    }
    if not tag or not stamp:
        return meta

    log = Path("logs/phase_f") / f"f3_{tag}_{stamp}_rollout.log"
    if not log.exists():
        return meta

    meta["run_log"] = str(log)
    try:
        lines = log.read_text(errors="ignore").splitlines()
    except Exception:
        return meta

    for line in lines:
        if "MANIFEST:" in line:
            meta["manifest"] = rel_repo_path_string(line.split("MANIFEST:", 1)[1].strip())
        if "D7 objective shadow csv=" in line:
            meta["d7_csv"] = rel_repo_path_string(line.split("D7 objective shadow csv=", 1)[1].strip())
        if "[TRACER] d5 log:" in line:
            meta["d5_log_dir"] = rel_repo_path_string(line.split("d5 log:", 1)[1].strip())

    return meta

def summarize_true_csv(p):
    tag, stamp = parse_tag_stamp(p)
    run_meta = parse_run_log_metadata(tag, stamp)
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None

    base_x = vals(rows, "base_x", "x", "pose_x")
    base_y = vals(rows, "base_y", "y", "pose_y")
    base_vx = vals(rows, "base_vx", "vx", "lin_vx")
    base_vy = vals(rows, "base_vy", "vy", "lin_vy")

    joint_abs_power = vals(rows, "joint_abs_power_sum", "joint_abs_power", "abs_power_sum")
    joint_abs_effort = vals(rows, "joint_abs_effort_sum", "joint_abs_effort", "abs_effort_sum")
    joint_sq_effort = vals(rows, "joint_sq_effort_sum", "joint_sq_effort", "sq_effort_sum")

    imu_wx = vals(rows, "imu_wx", "imu_ang_vel_x", "wx")
    imu_wy = vals(rows, "imu_wy", "imu_ang_vel_y", "wy")
    imu_wz = vals(rows, "imu_wz", "imu_ang_vel_z", "wz")
    imu_ax = vals(rows, "imu_ax", "imu_acc_x", "ax")
    imu_ay = vals(rows, "imu_ay", "imu_acc_y", "ay")
    imu_az = vals(rows, "imu_az", "imu_acc_z", "az")

    contact_count = vals(rows, "contact_count")
    contact_z = vals(rows, "contact_force_z_sum", "foot_contact_force_z_sum")
    contact_norm = vals(rows, "contact_force_norm_sum", "foot_contact_force_norm_sum")

    imu_ang_norm = []
    for a, b, c in zip(imu_wx, imu_wy, imu_wz):
        imu_ang_norm.append(math.sqrt(a*a + b*b + c*c))

    imu_acc_norm = []
    for a, b, c in zip(imu_ax, imu_ay, imu_az):
        imu_acc_norm.append(math.sqrt(a*a + b*b + c*c))

    base_dx = ""
    base_dy = ""
    if base_x:
        base_dx = base_x[-1] - base_x[0]
    if base_y:
        base_dy = base_y[-1] - base_y[0]

    row = {
        "tag": tag,
        "reset_y": infer_reset_y(tag),
        "manifest": run_meta.get("manifest") or latest_manifest_for_tag(tag),
        "run_log": run_meta.get("run_log", ""),
        "d5_log_dir": run_meta.get("d5_log_dir", ""),
        "d7_csv": run_meta.get("d7_csv", ""),
        "true_csv": str(p),
        "summary_csv": find_summary_csv(p),
        "n_rows": len(rows),
        "t_ros_start": rows[0].get("t_ros", rows[0].get("t", "")),
        "t_ros_end": rows[-1].get("t_ros", rows[-1].get("t", "")),
        "base_dx": base_dx,
        "base_dy": base_dy,
        "base_vx_mean": safe_mean(base_vx),
        "base_vx_std": safe_std(base_vx),
        "base_vy_abs_mean": safe_mean([abs(v) for v in base_vy]),
        "joint_abs_power_mean": safe_mean(joint_abs_power),
        "joint_abs_power_max": safe_max(joint_abs_power),
        "joint_abs_effort_mean": safe_mean(joint_abs_effort),
        "joint_sq_effort_mean": safe_mean(joint_sq_effort),
        "energy_proxy": safe_mean(joint_abs_power),
        "imu_ang_vel_norm_mean": safe_mean(imu_ang_norm),
        "imu_ang_vel_norm_max": safe_max(imu_ang_norm),
        "imu_acc_norm_mean": safe_mean(imu_acc_norm),
        "imu_acc_norm_max": safe_max(imu_acc_norm),
        "contact_count_mean": safe_mean(contact_count),
        "contact_force_z_sum_mean": safe_mean(contact_z),
        "contact_force_z_sum_max": safe_max(contact_z),
        "contact_force_norm_sum_mean": safe_mean(contact_norm),
        "contact_force_norm_sum_max": safe_max(contact_norm),
        "contact_load_proxy": safe_mean(contact_z),
    }
    return row

true_files = sorted(IN_DIR.glob("f3_*_true_metrics.csv"))
rows = []
for p in true_files:
    r = summarize_true_csv(p)
    if r:
        rows.append(r)

rows.sort(key=lambda r: (str(r["tag"]), str(r["true_csv"])))

if not rows:
    raise SystemExit("[TRACER] no f3_*_true_metrics.csv files found")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "tag", "reset_y", "manifest", "run_log", "d5_log_dir", "d7_csv", "true_csv", "summary_csv", "n_rows",
    "t_ros_start", "t_ros_end",
    "base_dx", "base_dy",
    "base_vx_mean", "base_vx_std", "base_vy_abs_mean",
    "joint_abs_power_mean", "joint_abs_power_max",
    "joint_abs_effort_mean", "joint_sq_effort_mean", "energy_proxy",
    "imu_ang_vel_norm_mean", "imu_ang_vel_norm_max",
    "imu_acc_norm_mean", "imu_acc_norm_max",
    "contact_count_mean",
    "contact_force_z_sum_mean", "contact_force_z_sum_max",
    "contact_force_norm_sum_mean", "contact_force_norm_sum_max",
    "contact_load_proxy",
]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F4 True Metric Training Table v0\n\n")
    f.write("This table is rebuilt directly from `f3_*_true_metrics.csv` files, so anchor sweep tags such as p045/p075/m045 are included even when a separate summary CSV is absent.\n\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- rows: `{len(rows)}`\n\n")
    f.write("## Rollouts\n\n")
    f.write("| tag | reset_y | n_rows | base_vx_mean | base_vy_abs_mean | joint_abs_power_mean | imu_ang_vel_norm_mean | contact_force_z_sum_mean |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r['tag']} | {r['reset_y']} | {r['n_rows']} | "
            f"{float(r['base_vx_mean'] or 0):.6f} | "
            f"{float(r['base_vy_abs_mean'] or 0):.6f} | "
            f"{float(r['joint_abs_power_mean'] or 0):.6f} | "
            f"{float(r['imu_ang_vel_norm_mean'] or 0):.6f} | "
            f"{float(r['contact_force_z_sum_mean'] or 0):.6f} |\n"
        )

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print(f"[TRACER] rows={len(rows)}")
print("[TRACER] tags:")
for r in rows:
    print(f"  {r['tag']}: reset_y={r['reset_y']} true_csv={r['true_csv']}")
