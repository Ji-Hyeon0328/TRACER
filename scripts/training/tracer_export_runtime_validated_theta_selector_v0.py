from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


def f(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def parse_theta(theta_text: str):
    vals = ast.literal_eval(theta_text)
    return [float(v) for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runtime-summary",
        default="artifacts/theta_safe_bank_runtime_v0/runtime_profile_summary.csv",
    )
    ap.add_argument(
        "--out",
        default="configs/runtime/tracer_runtime_validated_theta_selector_v0.json",
    )
    ap.add_argument("--min-default-feasible-frac", type=float, default=0.95)
    ap.add_argument("--min-profile-feasible-frac", type=float, default=0.80)
    args = ap.parse_args()

    rows = []
    with Path(args.runtime_summary).open(newline="") as fp:
        for row in csv.DictReader(fp):
            row["feasible_frac"] = f(row["feasible_frac"])
            row["runtime_score"] = f(row["runtime_score"])
            row["zmin_mean"] = f(row["zmin_mean"])
            row["distance_mean"] = f(row["distance_mean"])
            row["theta_norm_list"] = parse_theta(row["theta_norm"])
            rows.append(row)

    if not rows:
        raise SystemExit("No runtime summary rows found.")

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["feasible_frac"],
            r["runtime_score"],
            r["zmin_mean"],
            r["distance_mean"],
        ),
        reverse=True,
    )

    global_default = rows_sorted[0]

    profile_table = {}
    for row in rows_sorted:
        profile = row["profile"]
        if profile not in profile_table:
            profile_table[profile] = row

    selector = {
        "name": "tracer_runtime_validated_theta_selector_v0",
        "source_summary": str(args.runtime_summary),
        "selection_rule": {
            "global_default": (
                "highest feasible_frac, then runtime_score, zmin_mean, distance_mean"
            ),
            "profile_candidate": (
                "use requested profile only if feasible_frac >= min_profile_feasible_frac; "
                "otherwise fallback to global_default"
            ),
        },
        "thresholds": {
            "min_default_feasible_frac": args.min_default_feasible_frac,
            "min_profile_feasible_frac": args.min_profile_feasible_frac,
        },
        "global_default": {
            "profile": global_default["profile"],
            "preset": global_default["preset"],
            "theta_norm": global_default["theta_norm_list"],
            "feasible_frac": global_default["feasible_frac"],
            "runtime_score": global_default["runtime_score"],
            "zmin_mean": global_default["zmin_mean"],
            "distance_mean": global_default["distance_mean"],
        },
        "profiles": {},
    }

    for profile, row in profile_table.items():
        allowed = row["feasible_frac"] >= args.min_profile_feasible_frac
        selector["profiles"][profile] = {
            "profile": row["profile"],
            "preset": row["preset"],
            "theta_norm": row["theta_norm_list"],
            "feasible_frac": row["feasible_frac"],
            "runtime_score": row["runtime_score"],
            "zmin_mean": row["zmin_mean"],
            "distance_mean": row["distance_mean"],
            "runtime_allowed": bool(allowed),
            "fallback_profile": global_default["profile"] if not allowed else None,
            "fallback_preset": global_default["preset"] if not allowed else None,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selector, indent=2))

    print(f"[TRACER] wrote {out}")
    print()
    print("[TRACER] global default:")
    print(json.dumps(selector["global_default"], indent=2))
    print()
    print("[TRACER] profile table:")
    for p, r in selector["profiles"].items():
        print(
            f"- {p}: allowed={r['runtime_allowed']} "
            f"feasible={r['feasible_frac']:.3f} "
            f"score={r['runtime_score']:.4f} "
            f"preset={r['preset']}"
        )


if __name__ == "__main__":
    main()
