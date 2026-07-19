#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


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
    a = max(1e-6, float(a))
    b = max(1e-6, float(b))
    c = max(1e-6, float(c))
    s = a + b + c
    return a / s, b / s, c / s


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def load_rows(path):
    with Path(path).open(newline="") as fp:
        return list(csv.DictReader(fp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = load_rows(args.in_csv)
    if not rows:
        raise SystemExit("[ERROR] empty input")

    out_rows = []
    for r in rows:
        prior_m = f(r, "input_beta_motion_mean")
        prior_s = f(r, "input_beta_stability_mean")
        prior_e = f(r, "input_beta_energy_mean")

        motion_need = clamp(f(r, "motion_need"))
        stability_need = clamp(f(r, "stability_need"))
        energy_need = clamp(f(r, "energy_need"))

        lateral_bound = max(1e-6, f(r, "lateral_bound", 2.0))
        lateral_pressure = clamp(f(r, "rollout_max_abs_y") / lateral_bound)
        context_lateral_pressure = clamp(f(r, "max_abs_y_context") / lateral_bound)
        hold_pressure = clamp(f(r, "hold_drift") / 0.25)

        # v1 refinement:
        # - keep semantic prior as base
        # - stability reacts strongly to lateral deviation and hold settling errors
        # - energy reacts mildly to effort, avoiding global over-weighting
        # - motion increases when progress is poor, but successful goal-reaching prevents over-correction
        raw_m = prior_m + 0.35 * motion_need
        raw_s = (
            prior_s
            + 0.75 * stability_need
            + 0.35 * lateral_pressure
            + 0.20 * context_lateral_pressure
            + 0.45 * hold_pressure
        )
        raw_e = prior_e + 0.15 * energy_need

        bm, bs, be = normalize3(raw_m, raw_s, raw_e)

        rr = dict(r)
        rr["target_beta_motion_v0"] = r["target_beta_motion"]
        rr["target_beta_stability_v0"] = r["target_beta_stability"]
        rr["target_beta_energy_v0"] = r["target_beta_energy"]

        rr["target_beta_motion"] = f"{bm:.9f}"
        rr["target_beta_stability"] = f"{bs:.9f}"
        rr["target_beta_energy"] = f"{be:.9f}"

        rr["refine_lateral_pressure"] = f"{lateral_pressure:.9f}"
        rr["refine_context_lateral_pressure"] = f"{context_lateral_pressure:.9f}"
        rr["refine_hold_pressure"] = f"{hold_pressure:.9f}"
        out_rows.append(rr)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = list(out_rows[0].keys())
    with out_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    by_ctx = {}
    for r in out_rows:
        by_ctx.setdefault(r["context"], []).append(r)

    lines = []
    lines.append("# TRACER Phase-D7.0b Refined Beta Target Summary v1")
    lines.append("")
    lines.append(f"- input_csv: `{args.in_csv}`")
    lines.append(f"- output_csv: `{args.out_csv}`")
    lines.append(f"- samples: `{len(out_rows)}`")
    lines.append("")
    lines.append("## Context-level beta means")
    lines.append("")
    lines.append("| context | n | beta_m_v0 | beta_s_v0 | beta_e_v0 | beta_m_v1 | beta_s_v1 | beta_e_v1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for ctx in sorted(by_ctx):
        rs = by_ctx[ctx]
        lines.append(
            f"| {ctx} | {len(rs)} | "
            f"{mean(f(r,'target_beta_motion_v0') for r in rs):.3f} | "
            f"{mean(f(r,'target_beta_stability_v0') for r in rs):.3f} | "
            f"{mean(f(r,'target_beta_energy_v0') for r in rs):.3f} | "
            f"{mean(f(r,'target_beta_motion') for r in rs):.3f} | "
            f"{mean(f(r,'target_beta_stability') for r in rs):.3f} | "
            f"{mean(f(r,'target_beta_energy') for r in rs):.3f} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- v1 reduces global energy over-weighting compared with v0.")
    lines.append("- v1 increases stability beta for samples with larger lateral deviation or hold drift.")
    lines.append("- v1 is still a bootstrap heuristic target, intended for D7.0c Objective Selector training.")
    lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_md}")


if __name__ == "__main__":
    main()
