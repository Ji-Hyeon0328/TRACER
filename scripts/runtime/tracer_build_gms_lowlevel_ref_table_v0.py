#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


QUERY_SCRIPT = "scripts/runtime/tracer_gms_target_to_lowlevel_ref_v0.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/rollout_metrics/meta_gait_v0_gms_dataset.csv")
    ap.add_argument("--out_csv", default="data/rollout_metrics/meta_gait_v0_lowlevel_refs.csv")
    ap.add_argument("--out_json", default="data/rollout_metrics/meta_gait_v0_lowlevel_refs.json")
    ap.add_argument("--nominal_vx", type=float, default=0.10)
    ap.add_argument("--nominal_body_height", type=float, default=0.295)
    ap.add_argument("--nominal_clearance", type=float, default=0.030)
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.dataset).open()))
    out_rows = []
    json_rows = []

    for row in rows:
        name = row["name"]
        cmd = [
            "python3",
            QUERY_SCRIPT,
            "--dataset",
            args.dataset,
            "--name",
            name,
            "--nominal_vx",
            str(args.nominal_vx),
            "--nominal_body_height",
            str(args.nominal_body_height),
            "--nominal_clearance",
            str(args.nominal_clearance),
            "--json",
        ]
        ref = json.loads(subprocess.check_output(cmd, text=True))
        mg = ref["meta_gait_proposal"]
        ll = ref["lowlevel_reference"]

        mpc_array = ll["mpc_array"]

        out = {
            "name": name,
            "label": ref["label"],
            "semantic_mode": ref["semantic_mode"],
            "risk_level": ref["risk_level"],
            "recovery_needed": ref["recovery_needed"],
            "vx": mg["vx"],
            "body_height": mg["body_height"],
            "swing_clearance": mg["swing_clearance"],
            "impedance_scale": mg["impedance_scale"],
            "mpc_yaw_rate": mpc_array[0],
            "mpc_vx": mpc_array[1],
            "mpc_vy": mpc_array[2],
            "mpc_body_height": mpc_array[3],
            "mpc_swing_clearance": mpc_array[4],
            "mpc_enable": mpc_array[5],
            "mpc_array_json": json.dumps(mpc_array),
            "reward_mean": ref["source_metrics"]["reward_mean"],
            "directional_dx": ref["source_metrics"]["directional_dx"],
            "abs_final_dy": ref["source_metrics"]["abs_final_dy"],
            "done_count": ref["source_metrics"]["done_count"],
        }
        out_rows.append(out)
        json_rows.append(ref)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "label",
        "semantic_mode",
        "risk_level",
        "recovery_needed",
        "vx",
        "body_height",
        "swing_clearance",
        "impedance_scale",
        "mpc_yaw_rate",
        "mpc_vx",
        "mpc_vy",
        "mpc_body_height",
        "mpc_swing_clearance",
        "mpc_enable",
        "mpc_array_json",
        "reward_mean",
        "directional_dx",
        "abs_final_dy",
        "done_count",
    ]

    with out_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    out_json.write_text(json.dumps(json_rows, indent=2, sort_keys=True))

    print(f"wrote {out_csv} with {len(out_rows)} rows")
    print(f"wrote {out_json} with {len(json_rows)} rows")
    for r in out_rows:
        print(
            f"{r['name']:<38s} | "
            f"mode={r['semantic_mode']:<32s} | "
            f"mpc=[{r['mpc_yaw_rate']:.3f}, {r['mpc_vx']:.3f}, {r['mpc_vy']:.3f}, "
            f"{r['mpc_body_height']:.3f}, {r['mpc_swing_clearance']:.3f}, {r['mpc_enable']:.1f}]"
        )


if __name__ == "__main__":
    main()
