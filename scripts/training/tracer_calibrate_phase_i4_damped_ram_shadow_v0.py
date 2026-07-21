#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev


DEFAULT_PRIOR = [0.454269, 0.458988, 0.518588]  # I0 target means: slip, rough, sigma


def ff(x):
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def fmt(v):
    return "NA" if v is None else f"{v:.6f}"


def sat_rate(vals, lo=0.02, hi=0.98):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v <= lo or v >= hi) / len(vals)


def l1(a, b):
    if any(v is None for v in a + b):
        return None
    return sum(abs(a[i] - b[i]) for i in range(3))


def summarize(rows, group_name):
    out = {"group": group_name, "rows": len(rows)}

    for prefix in ["learned", "damped", "safe"]:
        cols = [f"{prefix}_slip", f"{prefix}_rough", f"{prefix}_sigma"]
        for c in cols:
            st = stats([ff(r.get(c)) for r in rows])
            out[f"{c}_mean"] = st["mean"]
            out[f"{c}_std"] = st["std"]
            out[f"{c}_min"] = st["min"]
            out[f"{c}_max"] = st["max"]
            out[f"{c}_sat"] = sat_rate([ff(r.get(c)) for r in rows])

    for c in ["legacy_slip", "legacy_rough", "legacy_sigma", "safe_l1_legacy", "learned_l1_legacy", "damped_l1_legacy"]:
        st = stats([ff(r.get(c)) for r in rows])
        out[f"{c}_mean"] = st["mean"]
        out[f"{c}_std"] = st["std"]
        out[f"{c}_max"] = st["max"]

    return out


ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out-csv", default="datasets/phase_i/i4_damped_ram_shadow_v0.csv")
ap.add_argument("--out-md", default="reports/phase_i/i4_damped_ram_shadow_summary_v0.md")
ap.add_argument("--out-json", default="models/phase_i/i4_damped_ram_postprocess_v0.json")
ap.add_argument("--gain", type=float, default=0.35)
ap.add_argument("--legacy-blend", type=float, default=0.35)
ap.add_argument("--floor", type=float, default=0.02)
ap.add_argument("--ceil", type=float, default=0.98)
args = ap.parse_args()

inp = Path(args.csv)
if not inp.exists():
    raise SystemExit(f"[TRACER] missing csv: {inp}")

rows = list(csv.DictReader(open(inp)))
if not rows:
    raise SystemExit(f"[TRACER] empty csv: {inp}")

prior = DEFAULT_PRIOR
out_rows = []

for r in rows:
    learned = [
        ff(r.get("rho_slip_pred")),
        ff(r.get("rho_rough_pred")),
        ff(r.get("sigma_pred")),
    ]
    legacy = [
        ff(r.get("legacy_slip")),
        ff(r.get("legacy_rough")),
        ff(r.get("legacy_sigma")),
    ]

    if any(v is None for v in learned):
        continue

    damped = [
        clamp(prior[i] + args.gain * (learned[i] - prior[i]), args.floor, args.ceil)
        for i in range(3)
    ]

    if all(v is not None for v in legacy):
        safe = [
            clamp((1.0 - args.legacy_blend) * legacy[i] + args.legacy_blend * damped[i], args.floor, args.ceil)
            for i in range(3)
        ]
    else:
        safe = damped[:]

    out = {
        "t_wall": r.get("t_wall", ""),
        "context": r.get("context", "unknown"),
        "x": r.get("x", ""),
        "y": r.get("y", ""),
        "observed_features": r.get("observed_features", ""),
        "imputed_features": r.get("imputed_features", ""),

        "learned_slip": f"{learned[0]:.8f}",
        "learned_rough": f"{learned[1]:.8f}",
        "learned_sigma": f"{learned[2]:.8f}",

        "damped_slip": f"{damped[0]:.8f}",
        "damped_rough": f"{damped[1]:.8f}",
        "damped_sigma": f"{damped[2]:.8f}",

        "safe_slip": f"{safe[0]:.8f}",
        "safe_rough": f"{safe[1]:.8f}",
        "safe_sigma": f"{safe[2]:.8f}",

        "legacy_slip": "" if legacy[0] is None else f"{legacy[0]:.8f}",
        "legacy_rough": "" if legacy[1] is None else f"{legacy[1]:.8f}",
        "legacy_sigma": "" if legacy[2] is None else f"{legacy[2]:.8f}",

        "learned_l1_legacy": "",
        "damped_l1_legacy": "",
        "safe_l1_legacy": "",
    }

    if all(v is not None for v in legacy):
        out["learned_l1_legacy"] = f"{l1(learned, legacy):.8f}"
        out["damped_l1_legacy"] = f"{l1(damped, legacy):.8f}"
        out["safe_l1_legacy"] = f"{l1(safe, legacy):.8f}"

    out_rows.append(out)

out_csv = Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)

fields = [
    "t_wall", "context", "x", "y", "observed_features", "imputed_features",
    "learned_slip", "learned_rough", "learned_sigma",
    "damped_slip", "damped_rough", "damped_sigma",
    "safe_slip", "safe_rough", "safe_sigma",
    "legacy_slip", "legacy_rough", "legacy_sigma",
    "learned_l1_legacy", "damped_l1_legacy", "safe_l1_legacy",
]

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

model = {
    "phase": "I4",
    "purpose": "Damped/calibrated RAM shadow postprocess",
    "input_log": str(inp),
    "output_csv": str(out_csv),
    "prior": {
        "source": "Phase-I0 target means",
        "rho_slip": prior[0],
        "rho_rough": prior[1],
        "sigma": prior[2],
    },
    "postprocess": {
        "damped": "prior + gain * (learned - prior)",
        "safe": "(1 - legacy_blend) * legacy + legacy_blend * damped when legacy is available, else damped",
        "gain": args.gain,
        "legacy_blend": args.legacy_blend,
        "floor": args.floor,
        "ceil": args.ceil,
    },
    "safe_claim": "Offline calibration only; not active RAM deployment.",
}

out_json = Path(args.out_json)
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(model, indent=2))

summaries = []
summaries.append(summarize(out_rows, "ALL"))
known = [r for r in out_rows if (r.get("context", "unknown") or "unknown") != "unknown"]
summaries.append(summarize(known, "KNOWN_CONTEXT_ONLY"))

by_ctx = defaultdict(list)
for r in out_rows:
    by_ctx[r.get("context", "unknown") or "unknown"].append(r)
for ctx in sorted(by_ctx):
    summaries.append(summarize(by_ctx[ctx], f"context:{ctx}"))

out_md = Path(args.out_md)
out_md.parent.mkdir(parents=True, exist_ok=True)

with open(out_md, "w") as f:
    f.write("# TRACER Phase-I4 Damped RAM Shadow v0\n\n")
    f.write("This applies an offline damped/safe postprocess to Phase-I2 learned RAM shadow outputs.\n\n")
    f.write(f"- input csv: `{inp}`\n")
    f.write(f"- output csv: `{out_csv}`\n")
    f.write(f"- postprocess json: `{out_json}`\n")
    f.write(f"- rows: `{len(out_rows)}`\n")
    f.write(f"- gain: `{args.gain}`\n")
    f.write(f"- legacy_blend: `{args.legacy_blend}`\n")
    f.write(f"- floor/ceil: `{args.floor}` / `{args.ceil}`\n\n")

    f.write("## Overall summary\n\n")
    f.write("| group | rows | learned L1 | damped L1 | safe L1 | learned sat slip/rough/sigma | damped sat slip/rough/sigma | safe sat slip/rough/sigma |\n")
    f.write("|---|---:|---:|---:|---:|---|---|---|\n")
    for s in summaries[:2]:
        f.write(
            f"| {s['group']} | {s['rows']} | "
            f"{fmt(s['learned_l1_legacy_mean'])} | {fmt(s['damped_l1_legacy_mean'])} | {fmt(s['safe_l1_legacy_mean'])} | "
            f"{fmt(s['learned_slip_sat'])}/{fmt(s['learned_rough_sat'])}/{fmt(s['learned_sigma_sat'])} | "
            f"{fmt(s['damped_slip_sat'])}/{fmt(s['damped_rough_sat'])}/{fmt(s['damped_sigma_sat'])} | "
            f"{fmt(s['safe_slip_sat'])}/{fmt(s['safe_rough_sat'])}/{fmt(s['safe_sigma_sat'])} |\n"
        )

    f.write("\n## By context\n\n")
    f.write("| context | rows | safe slip | safe rough | safe sigma | safe L1 | safe sat slip | safe sat rough | safe sat sigma |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in summaries[2:]:
        ctx = s["group"].replace("context:", "")
        f.write(
            f"| {ctx} | {s['rows']} | "
            f"{fmt(s['safe_slip_mean'])} | {fmt(s['safe_rough_mean'])} | {fmt(s['safe_sigma_mean'])} | "
            f"{fmt(s['safe_l1_legacy_mean'])} | "
            f"{fmt(s['safe_slip_sat'])} | {fmt(s['safe_rough_sat'])} | {fmt(s['safe_sigma_sat'])} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("- `learned` is the original I2 RAM output.\n")
    f.write("- `damped` keeps the learned direction but pulls it toward the I0 prior.\n")
    f.write("- `safe` additionally blends with legacy RAM when available.\n")
    f.write("- If `safe` sharply reduces saturation and L1 from legacy, it can be used as the next shadow runtime output. It should still not replace RAM in active control yet.\n")

print(f"[TRACER] wrote {out_csv}")
print(f"[TRACER] wrote {out_json}")
print(f"[TRACER] wrote {out_md}")
print(f"[TRACER] rows={len(out_rows)} known_context_rows={len(known)}")
