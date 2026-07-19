#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def f(row, key, default=0.0):
    try:
        v = row.get(key, default)
        if v in ("", None):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def load_rows(path):
    with Path(path).open(newline="") as fp:
        return list(csv.DictReader(fp))


def groupby(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r.get(key, "unknown")].append(r)
    return out


def beta_sum(row):
    return (
        f(row, "target_beta_motion") +
        f(row, "target_beta_stability") +
        f(row, "target_beta_energy")
    )


def fmt(x):
    return f"{x:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--top-k", type=int, default=12)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("[ERROR] empty dataset")

    lines = []
    lines.append("# TRACER Phase-D7.0b Objective Bootstrap Inspection v0")
    lines.append("")
    lines.append(f"- input_csv: `{args.csv}`")
    lines.append(f"- rows: `{len(rows)}`")
    lines.append("")

    sums = [beta_sum(r) for r in rows]
    lines.append("## Beta normalization check")
    lines.append("")
    lines.append(f"- beta_sum_min: `{min(sums):.9f}`")
    lines.append(f"- beta_sum_max: `{max(sums):.9f}`")
    lines.append(f"- beta_sum_mean: `{mean(sums):.9f}`")
    lines.append("")

    lines.append("## Context-level means")
    lines.append("")
    lines.append("| context | n | beta_m | beta_s | beta_e | motion_score | stability_score | energy_score | max_abs_y | hold_drift | effort_proxy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for ctx, rs in sorted(groupby(rows, "context").items()):
        lines.append(
            f"| {ctx} | {len(rs)} | "
            f"{mean(f(r,'target_beta_motion') for r in rs):.3f} | "
            f"{mean(f(r,'target_beta_stability') for r in rs):.3f} | "
            f"{mean(f(r,'target_beta_energy') for r in rs):.3f} | "
            f"{mean(f(r,'motion_score') for r in rs):.3f} | "
            f"{mean(f(r,'stability_score') for r in rs):.3f} | "
            f"{mean(f(r,'energy_score') for r in rs):.3f} | "
            f"{mean(f(r,'rollout_max_abs_y') for r in rs):.3f} | "
            f"{mean(f(r,'hold_drift') for r in rs):.3f} | "
            f"{mean(f(r,'effort_proxy') for r in rs):.3f} |"
        )

    lines.append("")
    lines.append("## Reset-y-level means")
    lines.append("")
    lines.append("| reset_y | n | beta_m | beta_s | beta_e | max_abs_y | hold_drift |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")

    for ry, rs in sorted(groupby(rows, "reset_y").items(), key=lambda kv: float(kv[0])):
        lines.append(
            f"| {float(ry):.2f} | {len(rs)} | "
            f"{mean(f(r,'target_beta_motion') for r in rs):.3f} | "
            f"{mean(f(r,'target_beta_stability') for r in rs):.3f} | "
            f"{mean(f(r,'target_beta_energy') for r in rs):.3f} | "
            f"{mean(f(r,'rollout_max_abs_y') for r in rs):.3f} | "
            f"{mean(f(r,'hold_drift') for r in rs):.3f} |"
        )

    def add_top(title, key, reverse=True):
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| rank | context | reset_y | trial | value | beta_m | beta_s | beta_e | max_abs_y | hold_drift | final_x | final_y | source |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        ranked = sorted(rows, key=lambda r: f(r, key), reverse=reverse)[: args.top_k]
        for i, r in enumerate(ranked, 1):
            lines.append(
                f"| {i} | {r.get('context','')} | {f(r,'reset_y'):.2f} | {int(f(r,'trial'))} | "
                f"{f(r,key):.3f} | "
                f"{f(r,'target_beta_motion'):.3f} | {f(r,'target_beta_stability'):.3f} | {f(r,'target_beta_energy'):.3f} | "
                f"{f(r,'rollout_max_abs_y'):.3f} | {f(r,'hold_drift'):.3f} | "
                f"{f(r,'final_x'):.3f} | {f(r,'final_y'):.3f} | "
                f"`{Path(r.get('source_summary','')).name}` |"
            )

    add_top("Highest hold drift samples", "hold_drift")
    add_top("Highest lateral deviation samples", "rollout_max_abs_y")
    add_top("Highest energy beta samples", "target_beta_energy")
    add_top("Lowest stability score samples", "stability_score", reverse=False)

    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    lines.append("- The bootstrap labels are normalized and usable as first beta targets.")
    lines.append("- Energy beta can become high even on successful fast motion because the current energy proxy directly penalizes vx/clearance/yaw effort.")
    lines.append("- For D7.0c training, a refined target should keep the semantic terrain prior while increasing stability more strongly for high lateral deviation or hold drift.")
    lines.append("- This is still a bootstrap objective selector target, not learned IRL yet.")
    lines.append("")

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[TRACER] wrote {out}")


if __name__ == "__main__":
    main()
