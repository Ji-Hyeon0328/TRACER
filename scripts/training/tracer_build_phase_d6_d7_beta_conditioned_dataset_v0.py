#!/usr/bin/env python3
import argparse
import bisect
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CTX_COLS = [
    "ctx_flat",
    "ctx_start_flat",
    "ctx_upslope",
    "ctx_rough",
    "ctx_downslope",
    "ctx_goal_flat",
    "ctx_unknown",
]


def fval(r, k, default=0.0):
    try:
        v = r.get(k, default)
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def read_csv(path):
    with Path(path).open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def read_manifest(path):
    rows = []
    with Path(path).open(newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def context_onehot(ctx):
    ctx = (ctx or "unknown").strip()
    out = {c: "0.0" for c in CTX_COLS}
    if ctx == "flat":
        out["ctx_flat"] = "1.0"
    elif ctx == "start_flat":
        out["ctx_start_flat"] = "1.0"
    elif ctx == "upslope":
        out["ctx_upslope"] = "1.0"
    elif ctx == "rough":
        out["ctx_rough"] = "1.0"
    elif ctx == "downslope":
        out["ctx_downslope"] = "1.0"
    elif ctx == "goal_flat":
        out["ctx_goal_flat"] = "1.0"
    else:
        out["ctx_unknown"] = "1.0"
    return out


def nearest_d7_row(t, d7_rows, d7_times, tolerance_s):
    i = bisect.bisect_left(d7_times, t)
    candidates = []
    if i < len(d7_rows):
        candidates.append(i)
    if i > 0:
        candidates.append(i - 1)
    if not candidates:
        return None, None

    best_i = min(candidates, key=lambda j: abs(d7_times[j] - t))
    dt = abs(d7_times[best_i] - t)
    if dt > tolerance_s:
        return None, None
    return d7_rows[best_i], dt


def mean_rows(rows, cols):
    out = {}
    for c in cols:
        vals = [fval(r, c) for r in rows if r.get(c, "") != ""]
        out[c] = sum(vals) / len(vals) if vals else 0.0
    return out


def build_from_d5_dir(manifest_path, manifest_row, tolerance_s):
    d5_dir = Path(manifest_row["d5_log_dir"])
    policy_csv = d5_dir / "policy_inputs_v0.csv"
    d7_csv = d5_dir / "d7_objective_selector_shadow_v0.csv"

    if not policy_csv.exists():
        raise FileNotFoundError(policy_csv)
    if not d7_csv.exists():
        raise FileNotFoundError(d7_csv)

    pol_rows = read_csv(policy_csv)
    d7_rows = read_csv(d7_csv)

    required_pol = [
        "t_wall", "context", "x", "y",
        "beta_motion", "beta_stability", "beta_energy",
        "ram_slip_proxy", "ram_roughness_proxy", "ram_sigma",
        "ref_seq", "ref_vx", "ref_yaw_rate", "ref_body_h", "ref_clearance", "ref_enable",
    ]
    required_d7 = [
        "t_wall",
        "pred_beta_motion", "pred_beta_stability", "pred_beta_energy",
        "raw_beta_motion", "raw_beta_stability", "raw_beta_energy",
        "actual_beta_motion", "actual_beta_stability", "actual_beta_energy",
    ]

    if not pol_rows:
        raise RuntimeError(f"empty policy csv: {policy_csv}")
    if not d7_rows:
        raise RuntimeError(f"empty D7 csv: {d7_csv}")

    missing_pol = [c for c in required_pol if c not in pol_rows[0]]
    missing_d7 = [c for c in required_d7 if c not in d7_rows[0]]
    if missing_pol:
        raise RuntimeError(f"missing policy columns in {policy_csv}: {missing_pol}")
    if missing_d7:
        raise RuntimeError(f"missing D7 columns in {d7_csv}: {missing_d7}")

    pol_rows = sorted(pol_rows, key=lambda r: fval(r, "t_wall"))
    d7_rows = sorted(d7_rows, key=lambda r: fval(r, "t_wall"))
    d7_times = [fval(r, "t_wall") for r in d7_rows]

    out = []
    dropped = 0

    for pr in pol_rows:
        t = fval(pr, "t_wall")
        d7r, dt = nearest_d7_row(t, d7_rows, d7_times, tolerance_s)
        if d7r is None:
            dropped += 1
            continue

        oh = context_onehot(pr["context"])

        row = {
            "source_manifest": str(manifest_path),
            "source_d5_log_dir": str(d5_dir),
            "trial": manifest_row.get("trial", ""),
            "world": manifest_row.get("world", ""),
            "t_wall": pr["t_wall"],
            "d7_t_wall": d7r["t_wall"],
            "dt_abs": f"{dt:.9f}",
            "context": pr["context"],

            **oh,

            "x": pr["x"],
            "y": pr["y"],

            # Trainer feature beta: D7 predicted beta.
            "beta_motion": d7r["pred_beta_motion"],
            "beta_stability": d7r["pred_beta_stability"],
            "beta_energy": d7r["pred_beta_energy"],

            # Diagnostics.
            "static_beta_motion": pr["beta_motion"],
            "static_beta_stability": pr["beta_stability"],
            "static_beta_energy": pr["beta_energy"],
            "raw_beta_motion": d7r["raw_beta_motion"],
            "raw_beta_stability": d7r["raw_beta_stability"],
            "raw_beta_energy": d7r["raw_beta_energy"],
            "actual_beta_motion": d7r["actual_beta_motion"],
            "actual_beta_stability": d7r["actual_beta_stability"],
            "actual_beta_energy": d7r["actual_beta_energy"],

            "ram_slip_proxy": pr["ram_slip_proxy"],
            "ram_roughness_proxy": pr["ram_roughness_proxy"],
            "ram_sigma": pr["ram_sigma"],

            # Original safe empirical references.
            "ref_seq": pr["ref_seq"],
            "ref_vx": pr["ref_vx"],
            "ref_yaw_rate": pr["ref_yaw_rate"],
            "ref_body_h": pr["ref_body_h"],
            "ref_clearance": pr["ref_clearance"],
            "ref_enable": pr["ref_enable"],

            # Trainer targets.
            "target_vx": pr["ref_vx"],
            "target_yaw_rate": pr["ref_yaw_rate"],
            "target_body_h": pr["ref_body_h"],
            "target_clearance": pr["ref_clearance"],
            "target_enable": pr["ref_enable"],
        }
        out.append(row)

    return out, {
        "manifest": str(manifest_path),
        "d5_dir": str(d5_dir),
        "before": len(pol_rows),
        "after": len(out),
        "dropped": dropped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--tolerance-s", type=float, default=0.25)
    args = ap.parse_args()

    all_rows = []
    source_stats = []

    for m in args.manifest:
        manifest_path = Path(m)
        for row in read_manifest(manifest_path):
            rows, stat = build_from_d5_dir(manifest_path, row, args.tolerance_s)
            all_rows.extend(rows)
            source_stats.append(stat)

    if not all_rows:
        raise RuntimeError("no rows built")

    out_cols = [
        "source_manifest", "source_d5_log_dir", "trial", "world",
        "t_wall", "d7_t_wall", "dt_abs",
        "context",
        "ctx_flat", "ctx_start_flat", "ctx_upslope", "ctx_rough",
        "ctx_downslope", "ctx_goal_flat", "ctx_unknown",
        "x", "y",
        "beta_motion", "beta_stability", "beta_energy",
        "static_beta_motion", "static_beta_stability", "static_beta_energy",
        "raw_beta_motion", "raw_beta_stability", "raw_beta_energy",
        "actual_beta_motion", "actual_beta_stability", "actual_beta_energy",
        "ram_slip_proxy", "ram_roughness_proxy", "ram_sigma",
        "ref_seq", "ref_vx", "ref_yaw_rate", "ref_body_h", "ref_clearance", "ref_enable",
        "target_vx", "target_yaw_rate", "target_body_h", "target_clearance", "target_enable",
    ]

    write_csv(args.out_csv, all_rows, out_cols)

    context_counts = Counter(r["context"] for r in all_rows)
    by_ctx = defaultdict(list)
    for r in all_rows:
        by_ctx[r["context"]].append(r)

    beta_by_context = {
        ctx: mean_rows(rows, ["beta_motion", "beta_stability", "beta_energy"])
        for ctx, rows in by_ctx.items()
    }
    target_by_context = {
        ctx: mean_rows(rows, ["target_vx", "target_yaw_rate", "target_body_h", "target_clearance", "target_enable"])
        for ctx, rows in by_ctx.items()
    }

    dt_vals = [fval(r, "dt_abs") for r in all_rows]
    summary = {
        "out_csv": args.out_csv,
        "rows": len(all_rows),
        "source_stats": source_stats,
        "context_counts": dict(context_counts),
        "beta_mean_by_context": beta_by_context,
        "target_mean_by_context": target_by_context,
        "dt_abs_mean": sum(dt_vals) / len(dt_vals),
        "dt_abs_max": max(dt_vals),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2))

    lines = []
    lines.append("# TRACER Phase-D6 D7-Beta-Conditioned Selector Dataset v0")
    lines.append("")
    lines.append(f"- rows: `{len(all_rows)}`")
    lines.append(f"- out_csv: `{args.out_csv}`")
    lines.append(f"- dt_abs_mean: `{summary['dt_abs_mean']:.6f}`")
    lines.append(f"- dt_abs_max: `{summary['dt_abs_max']:.6f}`")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    for s in source_stats:
        lines.append(f"- `{s['manifest']}` / `{s['d5_dir']}`: `{s['after']}/{s['before']}` rows kept, dropped `{s['dropped']}`")
    lines.append("")
    lines.append("## Context counts")
    lines.append("")
    lines.append("| context | count |")
    lines.append("|---|---:|")
    for k in sorted(context_counts):
        lines.append(f"| {k} | {context_counts[k]} |")
    lines.append("")
    lines.append("## D7 beta mean by context")
    lines.append("")
    lines.append("| context | beta_motion | beta_stability | beta_energy |")
    lines.append("|---|---:|---:|---:|")
    for k in sorted(beta_by_context):
        v = beta_by_context[k]
        lines.append(f"| {k} | {v['beta_motion']:.6f} | {v['beta_stability']:.6f} | {v['beta_energy']:.6f} |")
    lines.append("")
    lines.append("## Target reference mean by context")
    lines.append("")
    lines.append("| context | target_vx | target_yaw_rate | target_body_h | target_clearance | target_enable |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k in sorted(target_by_context):
        v = target_by_context[k]
        lines.append(f"| {k} | {v['target_vx']:.6f} | {v['target_yaw_rate']:.6f} | {v['target_body_h']:.6f} | {v['target_clearance']:.6f} | {v['target_enable']:.6f} |")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {args.out_csv}")
    print(f"[TRACER] wrote {args.out_json}")
    print(f"[TRACER] wrote {args.out_md}")


if __name__ == "__main__":
    main()
