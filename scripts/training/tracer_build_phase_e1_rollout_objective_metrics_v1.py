#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path
from statistics import mean

def ffloat(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def read_csv_rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def vals_col(rows, col):
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(col, "")))
        except Exception:
            pass
    return vals

def mean_col(rows, col, default=0.0):
    vals = vals_col(rows, col)
    return mean(vals) if vals else default

def last_col(rows, col, default=0.0):
    if not rows:
        return default
    return ffloat(rows[-1].get(col), default)

def parse_reset_y(d5_log_dir):
    p = Path(d5_log_dir) / "reset_y_offset_v0.log"
    if not p.exists():
        return 0.0

    text = p.read_text(errors="replace")

    m = re.search(r"\breset_y\s*=\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))

    m = re.search(r"state after y-offset:.*\by=([-+]?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))

    return 0.0

def infer_learned_dims(d5_log_dir):
    log_path = Path(d5_log_dir) / "gated_selector_dryrun_node.log"
    dims = {
        "learned_vx": "",
        "learned_yaw": "",
        "learned_body_h": "",
        "learned_clearance": "",
        "learned_enable": "",
    }
    if not log_path.exists():
        return dims

    text = log_path.read_text(errors="replace")
    for line in text.splitlines():
        if "learned_dims:" not in line:
            continue
        tail = line.split("learned_dims:", 1)[1]
        for part in tail.split(","):
            if "=" not in part:
                continue
            k, v = part.strip().split("=", 1)
            key = {
                "vx": "learned_vx",
                "yaw": "learned_yaw",
                "body_h": "learned_body_h",
                "clearance": "learned_clearance",
                "enable": "learned_enable",
            }.get(k.strip())
            if key:
                dims[key] = v.strip()
    return dims

def gate_metrics(gate_rows):
    if not gate_rows:
        return {
            "gate_rows": 0,
            "accept_rate": 0.0,
            "moving_accept_rate": 0.0,
            "hold_accept_rate": 0.0,
            "reject_missing_learned": 0,
            "reject_vx": 0,
            "reject_yaw": 0,
            "reject_clearance": 0,
            "reject_hold_phase": 0,
        }

    def is_accept(r):
        return str(r.get("gate_accept", "")).lower() in ("1", "true", "yes")

    accept = [r for r in gate_rows if is_accept(r)]
    moving = [
        r for r in gate_rows
        if r.get("context") != "goal_flat"
        and r.get("reject_reason") != "hold_phase_empirical"
    ]
    hold = [
        r for r in gate_rows
        if r.get("context") == "goal_flat"
        or r.get("reject_reason") == "hold_phase_empirical"
    ]

    reasons = {}
    for r in gate_rows:
        rr = r.get("reject_reason", "")
        if rr:
            reasons[rr] = reasons.get(rr, 0) + 1

    return {
        "gate_rows": len(gate_rows),
        "accept_rate": len(accept) / max(1, len(gate_rows)),
        "moving_accept_rate": sum(is_accept(r) for r in moving) / max(1, len(moving)),
        "hold_accept_rate": sum(is_accept(r) for r in hold) / max(1, len(hold)),
        "reject_missing_learned": reasons.get("missing_learned", 0),
        "reject_vx": reasons.get("vx", 0),
        "reject_yaw": reasons.get("yaw", 0),
        "reject_clearance": reasons.get("clearance", 0),
        "reject_hold_phase": reasons.get("hold_phase_empirical", 0),
    }

def count_contexts(rows):
    out = {}
    for r in rows:
        c = r.get("context", "unknown")
        out[c] = out.get(c, 0) + 1
    return out

def build_one(manifest_path):
    out_rows = []
    with open(manifest_path, newline="") as f:
        manifest_rows = list(csv.DictReader(f, delimiter="\t"))

    for mr in manifest_rows:
        d4 = mr.get("log_dir", "")
        d5 = mr.get("d5_log_dir", "")

        d7_rows = read_csv_rows(os.path.join(d5, "d7_objective_selector_shadow_v0.csv"))
        gate_rows = read_csv_rows(os.path.join(d5, "gated_selector_dryrun_v0.csv"))
        mlp_rows = read_csv_rows(os.path.join(d5, "mlp_selector_shadow_v0.csv"))

        y_vals = vals_col(d7_rows, "y")
        x_vals = vals_col(d7_rows, "x")

        final_x = x_vals[-1] if x_vals else 0.0
        final_y = y_vals[-1] if y_vals else 0.0
        path_max_abs_y = max((abs(v) for v in y_vals), default=0.0)
        path_mean_abs_y = mean([abs(v) for v in y_vals]) if y_vals else 0.0

        goal_x = ffloat(mr.get("goal_x"), 8.0)
        lateral_bound = ffloat(mr.get("lateral_bound"), 2.0)

        contexts = count_contexts(d7_rows)
        gm = gate_metrics(gate_rows)
        dims = infer_learned_dims(d5)
        reset_y = parse_reset_y(d5)

        success_proxy = 1 if final_x >= goal_x - 0.25 else 0
        goal_proxy = 1 if (d7_rows and d7_rows[-1].get("context") == "goal_flat") or success_proxy else 0
        out_lane_proxy = 1 if path_max_abs_y > lateral_bound else 0

        row = {
            "manifest": manifest_path,
            "trial": mr.get("trial", ""),
            "world": mr.get("world", ""),
            "goal_x": goal_x,
            "lateral_bound": lateral_bound,
            "hold_vx": mr.get("hold_vx", ""),
            "log_dir": d4,
            "d5_log_dir": d5,

            **dims,

            "d7_rows": len(d7_rows),
            "gate_rows": gm["gate_rows"],
            "mlp_rows": len(mlp_rows),

            "reset_y": reset_y,
            "final_x": final_x,
            "final_y": final_y,
            "success_proxy": success_proxy,
            "goal_proxy": goal_proxy,
            "out_lane_proxy": out_lane_proxy,

            "path_max_abs_y": path_max_abs_y,
            "path_mean_abs_y": path_mean_abs_y,

            "motion_score_mean": mean_col(d7_rows, "motion_score", 0.0),
            "stability_score_mean": mean_col(d7_rows, "stability_score", 0.0),
            "energy_score_mean": mean_col(d7_rows, "energy_score", 0.0),
            "effort_proxy_mean": mean_col(d7_rows, "effort_proxy", 0.0),
            "hold_drift_mean": mean_col(d7_rows, "hold_drift", 0.0),

            "pred_beta_motion_mean": mean_col(d7_rows, "pred_beta_motion", 0.0),
            "pred_beta_stability_mean": mean_col(d7_rows, "pred_beta_stability", 0.0),
            "pred_beta_energy_mean": mean_col(d7_rows, "pred_beta_energy", 0.0),

            "ram_slip_proxy_mean": mean_col(d7_rows, "ram_slip_proxy_mean", 0.0),
            "ram_roughness_proxy_mean": mean_col(d7_rows, "ram_roughness_proxy_mean", 0.0),
            "ram_sigma_mean": mean_col(d7_rows, "ram_sigma_mean", 0.0),

            "ref_vx_mean": mean_col(d7_rows, "ref_vx_mean", 0.0),
            "ref_yaw_rate_mean": mean_col(d7_rows, "ref_yaw_rate_mean", 0.0),
            "ref_clearance_mean": mean_col(d7_rows, "ref_clearance_mean", 0.0),

            **gm,

            "mean_err_vx": mean_col(mlp_rows, "err_vx", 0.0),
            "mean_err_yaw_rate": mean_col(mlp_rows, "err_yaw_rate", 0.0),
            "mean_err_clearance": mean_col(mlp_rows, "err_clearance", 0.0),

            "count_flat": contexts.get("flat", 0),
            "count_upslope": contexts.get("upslope", 0),
            "count_rough": contexts.get("rough", 0),
            "count_downslope": contexts.get("downslope", 0),
            "count_goal_flat": contexts.get("goal_flat", 0),
        }
        out_rows.append(row)

    return out_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = []
    for m in args.manifest:
        rows.extend(build_one(m))

    if not rows:
        raise SystemExit("No rows built")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with open(out_md, "w") as f:
        f.write("# TRACER Phase-E1 Rollout Objective Metrics v1\n\n")
        f.write("- fixes v0 reset_y parsing using reset_y_offset_v0.log\n")
        f.write("- computes path_max_abs_y/path_mean_abs_y directly from D7 y trajectory\n\n")
        f.write(f"- output_csv: `{out_csv}`\n")
        f.write(f"- num_rollouts: `{len(rows)}`\n")
        f.write("- manifests:\n")
        for m in args.manifest:
            f.write(f"  - `{m}`\n")

        f.write("\n## Rows\n\n")
        f.write("| idx | trial | reset_y | learned vx/yaw/clr | success | goal | out_lane | final_x | final_y | path_max_abs_y | path_mean_abs_y | moving_accept | motion | stability | energy | effort |\n")
        f.write("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for i, r in enumerate(rows):
            dims = f"{r['learned_vx']}/{r['learned_yaw']}/{r['learned_clearance']}"
            f.write(
                f"| {i} | {r['trial']} | {float(r['reset_y']):+.2f} | {dims} | "
                f"{r['success_proxy']} | {r['goal_proxy']} | {r['out_lane_proxy']} | "
                f"{float(r['final_x']):.3f} | {float(r['final_y']):+.3f} | "
                f"{float(r['path_max_abs_y']):.3f} | {float(r['path_mean_abs_y']):.3f} | "
                f"{float(r['moving_accept_rate']):.3f} | "
                f"{float(r['motion_score_mean']):.3f} | "
                f"{float(r['stability_score_mean']):.3f} | "
                f"{float(r['energy_score_mean']):.3f} | "
                f"{float(r['effort_proxy_mean']):.3f} |\n"
            )

    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_md}")

if __name__ == "__main__":
    main()
