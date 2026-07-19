#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict


CONTEXTS = ["flat", "start_flat", "upslope", "rough", "downslope", "goal_flat", "unknown"]


def fnum(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def context_onehot(ctx):
    ctx = ctx if ctx in CONTEXTS else "unknown"
    return {f"ctx_{c}": 1.0 if c == ctx else 0.0 for c in CONTEXTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-per-context", type=int, default=2000)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    per_context_count = Counter()
    per_trial_rows = []

    with manifest.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for mrow in reader:
            trial = mrow["trial"]
            d5_dir = Path(mrow["d5_log_dir"])
            d5_csv = d5_dir / "policy_inputs_v0.csv"

            if not d5_csv.is_file():
                print(f"[WARN] missing D5 log for trial={trial}: {d5_csv}")
                continue

            raw_rows = list(csv.DictReader(d5_csv.open()))
            used = 0

            for idx, r in enumerate(raw_rows):
                if args.stride > 1 and idx % args.stride != 0:
                    continue

                ctx = r.get("context", "unknown") or "unknown"
                if ctx not in CONTEXTS:
                    ctx = "unknown"

                if per_context_count[ctx] >= args.max_per_context:
                    continue

                x = fnum(r.get("x"))
                y = fnum(r.get("y"))
                ref_vx = fnum(r.get("ref_vx"))
                ref_yaw_rate = fnum(r.get("ref_yaw_rate"))
                ref_body_h = fnum(r.get("ref_body_h"))
                ref_clearance = fnum(r.get("ref_clearance"))
                ref_enable = fnum(r.get("ref_enable"))

                if x is None or y is None:
                    continue
                if ref_vx is None or ref_body_h is None or ref_clearance is None or ref_enable is None:
                    continue

                sample = {
                    "trial": int(trial),
                    "source_d5_log": str(d5_csv),
                    "t_wall": fnum(r.get("t_wall")),
                    "context": ctx,
                    "x": x,
                    "y": y,
                    "beta_motion": fnum(r.get("beta_motion")),
                    "beta_stability": fnum(r.get("beta_stability")),
                    "beta_energy": fnum(r.get("beta_energy")),
                    "ram_slip_proxy": fnum(r.get("ram_slip_proxy")),
                    "ram_roughness_proxy": fnum(r.get("ram_roughness_proxy")),
                    "ram_sigma": fnum(r.get("ram_sigma")),
                    "target_vx": ref_vx,
                    "target_yaw_rate": ref_yaw_rate,
                    "target_body_h": ref_body_h,
                    "target_clearance": ref_clearance,
                    "target_enable": ref_enable,
                }
                sample.update(context_onehot(ctx))

                samples.append(sample)
                per_context_count[ctx] += 1
                used += 1

            per_trial_rows.append({
                "trial": int(trial),
                "d5_csv": str(d5_csv),
                "raw_rows": len(raw_rows),
                "used_rows": used,
            })

    dataset_csv = Path(str(out_prefix) + "_dataset_v0.csv")
    summary_json = Path(str(out_prefix) + "_summary_v0.json")
    summary_md = Path(str(out_prefix) + "_summary_v0.md")

    fields = [
        "trial", "source_d5_log", "t_wall",
        "context", *[f"ctx_{c}" for c in CONTEXTS],
        "x", "y",
        "beta_motion", "beta_stability", "beta_energy",
        "ram_slip_proxy", "ram_roughness_proxy", "ram_sigma",
        "target_vx", "target_yaw_rate", "target_body_h", "target_clearance", "target_enable",
    ]

    with dataset_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in samples:
            w.writerow({k: s.get(k) for k in fields})

    summary = {
        "manifest": str(manifest),
        "dataset_csv": str(dataset_csv),
        "num_samples": len(samples),
        "stride": args.stride,
        "max_per_context": args.max_per_context,
        "context_counts": dict(per_context_count),
        "per_trial_rows": per_trial_rows,
        "feature_columns": [
            "context one-hot",
            "x", "y",
            "beta_motion", "beta_stability", "beta_energy",
            "ram_slip_proxy", "ram_roughness_proxy", "ram_sigma",
        ],
        "target_columns": [
            "target_vx", "target_yaw_rate", "target_body_h", "target_clearance", "target_enable",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    lines = []
    lines.append("# TRACER Phase-D5 Shadow Dataset v0")
    lines.append("")
    lines.append(f"- manifest: `{manifest}`")
    lines.append(f"- dataset_csv: `{dataset_csv}`")
    lines.append(f"- num_samples: `{len(samples)}`")
    lines.append(f"- stride: `{args.stride}`")
    lines.append("")
    lines.append("## Context counts")
    lines.append("")
    lines.append("| context | samples |")
    lines.append("|---|---:|")
    for k, v in per_context_count.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Per-trial rows")
    lines.append("")
    lines.append("| trial | raw_rows | used_rows | d5_csv |")
    lines.append("|---:|---:|---:|---|")
    for r in per_trial_rows:
        lines.append(f"| {r['trial']} | {r['raw_rows']} | {r['used_rows']} | `{r['d5_csv']}` |")

    summary_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {dataset_csv}")
    print(f"[TRACER] wrote {summary_json}")
    print(f"[TRACER] wrote {summary_md}")
    print(f"[TRACER] num_samples={len(samples)}")
    print(f"[TRACER] context_counts={dict(per_context_count)}")


if __name__ == "__main__":
    main()
