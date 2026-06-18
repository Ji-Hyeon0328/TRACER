#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def truthy(v):
    return str(v).lower() in ("true", "1", "yes", "y")


def fmt(v):
    try:
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)
    except Exception:
        return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/style_missions/style_mission_summary.csv")
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"[ERROR] missing summary CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Skip invalid runs if they ever appear in CSV.
    if "run_id" in df.columns:
        df = df[~df["run_id"].astype(str).str.startswith("INVALID_")]

    # Normalize possible column names.
    rename = {
        "vx_cmd": "vx",
        "body_height_cmd": "height",
        "clearance_cmd": "clearance",
        "net displacement": "net_disp",
        "net_disp_m": "net_disp",
        "path_efficiency": "path_eff",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    required = ["terrain", "style", "status", "success", "duration", "path", "net_disp", "path_eff"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise SystemExit(f"[ERROR] missing required columns: {missing_required}\nAvailable: {list(df.columns)}")

    if args.success_only:
        df = df[df["success"].map(truthy)]

    numeric_cols = ["duration", "path", "net_disp", "path_eff", "vx", "height", "clearance"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_spec = {
        "n": ("run_id", "count") if "run_id" in df.columns else ("style", "count"),
        "success_rate": ("success", lambda x: sum(truthy(v) for v in x) / max(len(x), 1)),
        "duration_mean": ("duration", "mean"),
        "duration_std": ("duration", "std"),
        "path_mean": ("path", "mean"),
        "net_disp_mean": ("net_disp", "mean"),
        "path_eff_mean": ("path_eff", "mean"),
        "path_eff_std": ("path_eff", "std"),
    }

    # Optional command columns.
    if "vx" in df.columns:
        agg_spec["vx"] = ("vx", "mean")
    if "height" in df.columns:
        agg_spec["height"] = ("height", "mean")
    if "clearance" in df.columns:
        agg_spec["clearance"] = ("clearance", "mean")

    group = (
        df.groupby(["terrain", "style"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["terrain", "style"])
    )

    if args.out:
        out_csv = Path(args.out)
    else:
        suffix = "_success_only" if args.success_only else ""
        out_csv = csv_path.with_name(f"style_mission_group_summary{suffix}.csv")
    out_md = out_csv.with_suffix(".md")

    group.to_csv(out_csv, index=False)

    # Manual markdown so we do not depend on tabulate.
    cols = list(group.columns)
    lines = []
    title = "# TRACER Style Group Summary"
    if args.success_only:
        title += " Success Only"
    lines.append(title)
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in group.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")

    out_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] rows: {len(df)}")
    print(f"[TRACER] groups: {len(group)}")
    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_md}")
    print()
    print(out_md.read_text())


if __name__ == "__main__":
    main()
