#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


TARGETS = [
    "target_vx",
    "target_yaw_rate",
    "target_body_h",
    "target_clearance",
    "target_enable",
]


def metric_block_ridge(j):
    return j["metrics"]


def metric_block_mlp(j):
    return j["test_metrics"]


def fmt(x):
    return "NA" if x is None else f"{x:.8f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge-json", required=True)
    ap.add_argument("--mlp-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    ridge = json.loads(Path(args.ridge_json).read_text())
    mlp = json.loads(Path(args.mlp_json).read_text())
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)

    r = metric_block_ridge(ridge)
    m = metric_block_mlp(mlp)

    lines = []
    lines.append("# TRACER Phase-D6 Ridge vs MLP Selector Comparison v0")
    lines.append("")
    lines.append(f"- ridge_json: `{args.ridge_json}`")
    lines.append(f"- mlp_json: `{args.mlp_json}`")
    lines.append("")
    lines.append("| target | ridge_test_rmse | mlp_test_rmse | ridge_test_mae | mlp_test_mae |")
    lines.append("|---|---:|---:|---:|---:|")

    for t in TARGETS:
        lines.append(
            f"| {t} | "
            f"{fmt(r[t]['test_rmse'])} | {fmt(m[t]['rmse'])} | "
            f"{fmt(r[t]['test_mae'])} | {fmt(m[t]['mae'])} |"
        )

    out.write_text("\n".join(lines) + "\n")
    print(f"[TRACER] wrote {out}")


if __name__ == "__main__":
    main()
