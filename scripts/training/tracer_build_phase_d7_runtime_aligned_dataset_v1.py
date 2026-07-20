#!/usr/bin/env python3
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


CONTEXT_PRIOR_BETA = {
    "flat":       (0.45, 0.25, 0.30),
    "start_flat": (0.45, 0.25, 0.30),
    "rough":      (0.25, 0.55, 0.20),
    "upslope":    (0.35, 0.40, 0.25),
    "downslope":  (0.25, 0.60, 0.15),
    "goal_flat":  (0.20, 0.50, 0.30),
    "unknown":    (0.33, 0.34, 0.33),
}

FEATURES = [
    "reset_y",
    "y_mean",
    "mean_abs_y_context",
    "max_abs_y_context",
    "ram_slip_proxy_mean",
    "ram_roughness_proxy_mean",
    "ram_sigma_mean",
    "motion_score",
    "stability_score",
    "energy_score",
    "effort_proxy",
    "hold_drift",
    "rollout_max_abs_y",
    "rollout_mean_abs_y",
]


def f(row, key, default=0.0):
    try:
        v = row.get(key, default)
        if v in ("", None):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def normalize3(a, b, c):
    vals = [max(1e-9, float(a)), max(1e-9, float(b)), max(1e-9, float(c))]
    s = sum(vals)
    return vals[0] / s, vals[1] / s, vals[2] / s


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def load_manifest(path):
    rows = []
    with Path(path).open(newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def load_csv(path):
    with Path(path).open(newline="") as fp:
        return list(csv.DictReader(fp))


def compute_runtime_target(row, lateral_bound):
    ctx = row.get("context", "unknown").strip() or "unknown"
    prior = CONTEXT_PRIOR_BETA.get(ctx, CONTEXT_PRIOR_BETA["unknown"])

    motion_score = clamp(f(row, "motion_score"))
    stability_score = clamp(f(row, "stability_score"))
    energy_score = clamp(f(row, "energy_score"))

    motion_need = 1.0 - motion_score
    stability_need = 1.0 - stability_score
    energy_need = 1.0 - energy_score

    lateral_pressure = clamp(f(row, "rollout_max_abs_y") / max(lateral_bound, 1e-9))
    context_lateral_pressure = clamp(f(row, "max_abs_y_context") / max(lateral_bound, 1e-9))
    hold_pressure = clamp(f(row, "hold_drift") / 0.25)

    # Same semantic idea as D7.0b v1, but applied row-wise to runtime/window features.
    raw_m = prior[0] + 0.35 * motion_need
    raw_s = (
        prior[1]
        + 0.75 * stability_need
        + 0.35 * lateral_pressure
        + 0.20 * context_lateral_pressure
        + 0.45 * hold_pressure
    )
    raw_e = prior[2] + 0.15 * energy_need

    return normalize3(raw_m, raw_s, raw_e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-per-context", type=int, default=700)
    ap.add_argument("--lateral-bound", type=float, default=2.0)
    ap.add_argument("--include-unknown", action="store_true")
    args = ap.parse_args()

    manifest_rows = load_manifest(args.manifest)
    samples = []
    ctx_count = Counter()

    for mr in manifest_rows:
        trial = mr.get("trial", "")
        d5_dir = Path(mr["d5_log_dir"])
        csv_path = d5_dir / "d7_objective_selector_shadow_v0.csv"

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        rows = load_csv(csv_path)
        for idx, row in enumerate(rows):
            if args.stride > 1 and idx % args.stride != 0:
                continue

            ctx = row.get("context", "unknown").strip() or "unknown"
            if ctx == "unknown" and not args.include_unknown:
                continue

            if ctx_count[ctx] >= args.max_per_context:
                continue

            bm, bs, be = compute_runtime_target(row, args.lateral_bound)

            out = {
                "source_manifest": args.manifest,
                "source_d5_dir": str(d5_dir),
                "source_csv": str(csv_path),
                "source_trial": trial,
                "source_row_index": idx,
                "context": ctx,
                "target_beta_motion": f"{bm:.9f}",
                "target_beta_stability": f"{bs:.9f}",
                "target_beta_energy": f"{be:.9f}",
                "runtime_pred_beta_motion": row.get("pred_beta_motion", ""),
                "runtime_pred_beta_stability": row.get("pred_beta_stability", ""),
                "runtime_pred_beta_energy": row.get("pred_beta_energy", ""),
                "runtime_raw_beta_motion": row.get("raw_beta_motion", ""),
                "runtime_raw_beta_stability": row.get("raw_beta_stability", ""),
                "runtime_raw_beta_energy": row.get("raw_beta_energy", ""),
            }

            for k in FEATURES:
                out[k] = f"{f(row, k):.9f}"

            samples.append(out)
            ctx_count[ctx] += 1

    if not samples:
        raise SystemExit("[ERROR] no runtime-aligned samples generated")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = list(samples[0].keys())
    with out_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)

    by = defaultdict(list)
    for s in samples:
        by[s["context"]].append(s)

    lines = []
    lines.append("# TRACER Phase-D7.1b Runtime-Aligned Objective Dataset Summary v1")
    lines.append("")
    lines.append(f"- manifest: `{args.manifest}`")
    lines.append(f"- out_csv: `{args.out_csv}`")
    lines.append(f"- samples: `{len(samples)}`")
    lines.append(f"- stride: `{args.stride}`")
    lines.append(f"- max_per_context: `{args.max_per_context}`")
    lines.append(f"- lateral_bound: `{args.lateral_bound}`")
    lines.append("")
    lines.append("## Context counts and beta means")
    lines.append("")
    lines.append("| context | n | beta_m | beta_s | beta_e | motion_score | stability_score | energy_score | rollout_max_abs_y | hold_drift |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for ctx in sorted(by):
        rs = by[ctx]
        def m(k):
            return mean(float(r[k]) for r in rs)
        lines.append(
            f"| {ctx} | {len(rs)} | "
            f"{m('target_beta_motion'):.3f} | "
            f"{m('target_beta_stability'):.3f} | "
            f"{m('target_beta_energy'):.3f} | "
            f"{m('motion_score'):.3f} | "
            f"{m('stability_score'):.3f} | "
            f"{m('energy_score'):.3f} | "
            f"{m('rollout_max_abs_y'):.3f} | "
            f"{m('hold_drift'):.3f} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This dataset uses the same online/window feature definitions produced by the D7 runtime shadow node.")
    lines.append("- Targets are runtime-aligned bootstrap beta labels, not preference-based IRL labels yet.")
    lines.append("- This is intended to train the first runtime-aligned Objective Selector model.")
    lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_md}")
    print(f"[TRACER] samples={len(samples)}")
    print("[TRACER] context counts:")
    for ctx, n in sorted(ctx_count.items()):
        print(f"  {ctx}: {n}")


if __name__ == "__main__":
    main()
