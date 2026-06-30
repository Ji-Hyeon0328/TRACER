from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def f(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def read_rows(path: Path):
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def theta_key(theta_text: str) -> str:
    try:
        vals = ast.literal_eval(theta_text)
        return json.dumps([round(float(v), 6) for v in vals])
    except Exception:
        return theta_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="data/rollouts/safe_bank_runtime_v0",
    )
    ap.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Explicit runtime rollout directory to include. Can be passed multiple times.",
    )
    ap.add_argument(
        "--out-dir",
        default="artifacts/theta_safe_bank_runtime_v0",
    )
    ap.add_argument("--min-episodes", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)

    if args.run_dir:
        summary_paths = []
        for rd in args.run_dir:
            rd_path = Path(rd)
            summary_paths.append(rd_path / "theta_sweep_summary.csv")
    else:
        summary_paths = sorted(root.glob("safe_bank_runtime_v0_*/theta_sweep_summary.csv"))

    for summary in summary_paths:
        if not summary.exists():
            print(f"[TRACER][WARN] missing summary: {summary}")
            continue
        run_dir = summary.parent
        for row in read_rows(summary):
            profile = row.get("profile", "unknown")
            preset = row.get("preset", "unknown")
            theta = theta_key(row.get("theta_norm", "[]"))
            groups[(profile, preset, theta)].append((run_dir.name, row))

    out_rows = []
    jsonl_rows = []

    for (profile, preset, theta), items in groups.items():
        if len(items) < args.min_episodes:
            continue

        zs = [f(r["proprio_base_z_min"]) for _, r in items]
        dists = [f(r["distance_xy_proxy"]) for _, r in items]
        rolls = [f(r["proprio_roll_abs_max"]) for _, r in items]
        pitches = [f(r["proprio_pitch_abs_max"]) for _, r in items]
        efforts = [f(r.get("command_effort_proxy", "nan")) for _, r in items]

        feasible_flags = []
        for _, r in items:
            z_ok = f(r["proprio_base_z_min"]) >= 0.22
            roll_ok = f(r["proprio_roll_abs_max"]) <= 0.20
            pitch_ok = f(r["proprio_pitch_abs_max"]) <= 0.25
            valid = str(r.get("valid_data", "")).lower() == "true"
            stable = str(r.get("proprio_base_height_stable", "")).lower() == "true"
            feasible_flags.append(bool(valid and stable and z_ok and roll_ok and pitch_ok))

        feasible_frac = sum(feasible_flags) / max(1, len(feasible_flags))

        # Runtime score: deliberately simple and interpretable.
        # Prioritize feasibility first, then height margin, then motion, then low effort.
        z_margin = max(0.0, min(1.0, (mean(zs) - 0.18) / 0.10))
        motion_proxy = max(0.0, min(1.0, mean(dists) / 0.08))
        effort_proxy = mean(efforts) if efforts and efforts[0] == efforts[0] else 0.05
        energy_proxy = max(0.0, min(1.0, 1.0 - effort_proxy))

        runtime_score = (
            0.55 * feasible_frac
            + 0.20 * z_margin
            + 0.15 * motion_proxy
            + 0.10 * energy_proxy
        )

        row_out = {
            "profile": profile,
            "preset": preset,
            "theta_norm": theta,
            "episodes": len(items),
            "feasible_frac": feasible_frac,
            "runtime_score": runtime_score,
            "zmin_mean": mean(zs),
            "zmin_std": pstdev(zs) if len(zs) > 1 else 0.0,
            "distance_mean": mean(dists),
            "roll_max_mean": mean(rolls),
            "pitch_max_mean": mean(pitches),
            "effort_mean": effort_proxy,
            "runs": ";".join(sorted(set(run for run, _ in items))),
        }

        out_rows.append(row_out)
        jsonl_rows.append(row_out)

    out_rows.sort(
        key=lambda r: (
            f(r["feasible_frac"]),
            f(r["runtime_score"]),
            f(r["zmin_mean"]),
            f(r["distance_mean"]),
        ),
        reverse=True,
    )

    out_csv = out_dir / "runtime_profile_summary.csv"
    out_jsonl = out_dir / "runtime_profile_summary.jsonl"

    if out_rows:
        with out_csv.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)

        with out_jsonl.open("w") as fp:
            for row in out_rows:
                fp.write(json.dumps(row) + "\n")

    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_jsonl}")
    print()
    print("rank,profile,preset,episodes,feasible_frac,runtime_score,zmin_mean,distance_mean")
    for i, r in enumerate(out_rows[:20], 1):
        print(
            f"{i},{r['profile']},{r['preset']},{r['episodes']},"
            f"{r['feasible_frac']:.3f},{r['runtime_score']:.4f},"
            f"{r['zmin_mean']:.4f},{r['distance_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
