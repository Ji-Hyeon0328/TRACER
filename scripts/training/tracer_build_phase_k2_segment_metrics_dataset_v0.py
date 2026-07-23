#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


KNOWN_CONTEXTS = [
    "flat",
    "start_flat",
    "rough",
    "upslope",
    "downslope",
    "goal_flat",
    "unknown",
]


def ff(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def get_any(row, names, default=""):
    for n in names:
        if n in row and row[n] != "":
            return row[n]
    return default


def norm_context(x):
    x = (x or "unknown").strip()
    if x in KNOWN_CONTEXTS:
        return x
    return "unknown"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize_xy(rows):
    valid = []
    for r in rows:
        x = ff(get_any(r, ["x", "pos_x", "base_x"]))
        y = ff(get_any(r, ["y", "pos_y", "base_y"]))
        if x is None or y is None:
            continue
        valid.append((x, y, r))

    if not valid:
        return None

    xs = [v[0] for v in valid]
    ys = [v[1] for v in valid]

    return {
        "rows": len(rows),
        "valid_xy": len(valid),
        "x_start": xs[0],
        "x_end": xs[-1],
        "x_min": min(xs),
        "x_max": max(xs),
        "y_start": ys[0],
        "y_end": ys[-1],
        "y_min": min(ys),
        "y_max": max(ys),
        "delta_y": ys[-1] - ys[0],
        "abs_delta_y": abs(ys[-1] - ys[0]),
        "max_abs_y": max(abs(y) for y in ys),
        "mean_abs_y": sum(abs(y) for y in ys) / len(ys),
    }


def parse_j7_by_context(j7_csv):
    out = defaultdict(lambda: {
        "j7_rows": 0,
        "j7_accepted": 0,
        "j7_projected": 0,
        "j7_empirical": 0,
        "j7_failsafe": 0,
    })

    if not j7_csv or j7_csv == "NA" or not Path(j7_csv).exists():
        return out

    rows = read_csv(j7_csv)
    for r in rows:
        ctx = norm_context(get_any(r, ["context", "terrain_context", "label"]))
        reason = get_any(r, ["reason", "decision_reason"])
        source = get_any(r, ["source", "output_source"])

        out[ctx]["j7_rows"] += 1
        if reason == "accepted_guarded" or get_any(r, ["accepted"]) in ("1", "true", "True"):
            out[ctx]["j7_accepted"] += 1
        if source == "projected":
            out[ctx]["j7_projected"] += 1
        elif source == "empirical":
            out[ctx]["j7_empirical"] += 1
        elif source == "failsafe_hold":
            out[ctx]["j7_failsafe"] += 1

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    summary_rows = read_csv(args.summary_csv)
    out_rows = []

    for sr in summary_rows:
        status = sr.get("status", "ok")
        if status != "ok":
            continue

        mode = sr["mode"]
        idx = sr["idx"]
        d5_log_dir = sr.get("d5_log_dir", "")
        j7_csv = sr.get("j7_csv", "NA")
        policy_csv = Path(d5_log_dir) / "policy_inputs_v0.csv"

        if not policy_csv.exists():
            continue

        policy_rows = read_csv(policy_csv)
        j7_ctx = parse_j7_by_context(j7_csv)

        grouped = defaultdict(list)
        for r in policy_rows:
            ctx = norm_context(get_any(r, ["context", "terrain_context", "label"]))
            grouped[ctx].append(r)

        # per-context aggregate
        for ctx, rows in grouped.items():
            s = summarize_xy(rows)
            if s is None:
                continue

            j = j7_ctx[ctx]
            out_rows.append({
                "mode": mode,
                "idx": idx,
                "context": ctx,
                "scope": "context_all",
                **{k: f"{v:.6f}" if isinstance(v, float) else v for k, v in s.items()},
                "j7_rows": j["j7_rows"],
                "j7_accepted": j["j7_accepted"],
                "j7_projected": j["j7_projected"],
                "j7_empirical": j["j7_empirical"],
                "j7_failsafe": j["j7_failsafe"],
                "run_max_abs_y": sr.get("max_abs_y", ""),
                "run_mean_abs_y": sr.get("mean_abs_y", ""),
                "manifest": sr.get("manifest", ""),
                "d5_log_dir": d5_log_dir,
                "j7_csv": j7_csv,
            })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "mode", "idx", "context", "scope",
        "rows", "valid_xy",
        "x_start", "x_end", "x_min", "x_max",
        "y_start", "y_end", "y_min", "y_max",
        "delta_y", "abs_delta_y", "max_abs_y", "mean_abs_y",
        "j7_rows", "j7_accepted", "j7_projected", "j7_empirical", "j7_failsafe",
        "run_max_abs_y", "run_mean_abs_y",
        "manifest", "d5_log_dir", "j7_csv",
    ]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # Markdown summary
    groups = defaultdict(list)
    for r in out_rows:
        groups[(r["mode"], r["context"])].append(r)

    lines = []
    lines.append("# TRACER Phase-K2 Segment Metrics Dataset Summary v0")
    lines.append("")
    lines.append(f"- source summary: `{args.summary_csv}`")
    lines.append(f"- output dataset: `{args.out_csv}`")
    lines.append("")
    lines.append("## Mode/context segment summary")
    lines.append("")
    lines.append("| mode | context | n | max_abs_y mean | mean_abs_y mean | abs_delta_y mean | j7 accepted mean | j7 projected mean |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    def mean(vals):
        vals = [float(v) for v in vals]
        return sum(vals) / len(vals) if vals else float("nan")

    order = [
        ("baseline", "flat"),
        ("j19_profile", "flat"),
        ("baseline", "rough"),
        ("j19_profile", "rough"),
        ("baseline", "upslope"),
        ("j19_profile", "upslope"),
        ("baseline", "downslope"),
        ("j19_profile", "downslope"),
        ("baseline", "goal_flat"),
        ("j19_profile", "goal_flat"),
        ("baseline", "unknown"),
        ("j19_profile", "unknown"),
    ]

    used = set()
    for key in order + sorted(groups.keys()):
        if key in used or key not in groups:
            continue
        used.add(key)
        rs = groups[key]
        mode, ctx = key
        lines.append(
            f"| {mode} | {ctx} | {len(rs)} | "
            f"{mean([r['max_abs_y'] for r in rs]):.6f} | "
            f"{mean([r['mean_abs_y'] for r in rs]):.6f} | "
            f"{mean([r['abs_delta_y'] for r in rs]):.6f} | "
            f"{mean([r['j7_accepted'] for r in rs]):.2f} | "
            f"{mean([r['j7_projected'] for r in rs]):.2f} |"
        )

    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- If j19_profile improves flat but worsens later contexts, the active flat command may perturb transition state.")
    lines.append("- If j19_profile and baseline are similar in every non-flat context, J19 can be kept as a neutral safe-routing scaffold.")
    lines.append("- If flat is improved and non-flat is not worse, J19 flat conservative routing becomes a stronger safe candidate.")
    lines.append("- This dataset is intended for Phase-K3 conservative action selector / risk analysis.")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER:K2] wrote {args.out_csv}")
    print(f"[TRACER:K2] wrote {args.out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
