#!/usr/bin/env python3

import argparse
import csv
from collections import Counter
from pathlib import Path


FIELDS = [
    "source_phase",
    "mode",
    "idx",
    "status",
    "goal_reached",
    "final_x",
    "final_y",
    "max_abs_y",
    "mean_abs_y",
    "risk_score_lower_better",
    "decision",
    "manifest",
    "d5_log_dir",
    "j7_csv",
    "run_dir",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--source-phase",
        default="L4A_PILOT",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open(newline="") as f:
        source_rows = list(csv.DictReader(f))

    output_rows = []
    skipped = []

    for row in source_rows:
        status = str(row.get("status", "")).strip()

        if status != "ok":
            skipped.append(
                (
                    row.get("mode", ""),
                    row.get("idx", ""),
                    status,
                )
            )
            continue

        output_rows.append({
            "source_phase": args.source_phase,
            "mode": row.get("mode", ""),
            "idx": row.get("idx", ""),
            "status": status,
            "goal_reached": row.get("goal_reached", ""),
            "final_x": row.get("final_x", ""),
            "final_y": row.get("final_y", ""),
            "max_abs_y": row.get("max_abs_y", ""),
            "mean_abs_y": row.get("mean_abs_y", ""),
            "risk_score_lower_better": "",
            "decision": "pilot_unlabeled",
            "manifest": row.get("manifest", ""),
            "d5_log_dir": row.get("d5_log_dir", ""),
            "j7_csv": row.get("j7_csv", ""),
            "run_dir": row.get("run_dir", ""),
        })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("[L4B] source rows =", len(source_rows))
    print("[L4B] adapted rows =", len(output_rows))
    print("[L4B] skipped =", skipped)
    print(
        "[L4B] modes =",
        dict(
            Counter(
                row["mode"]
                for row in output_rows
            )
        ),
    )
    print("[L4B] output =", output_path)


if __name__ == "__main__":
    main()
