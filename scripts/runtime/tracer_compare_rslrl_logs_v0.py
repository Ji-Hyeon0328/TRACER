from __future__ import annotations

import argparse
import re
from pathlib import Path


PATTERNS = {
    "iteration": re.compile(r"Learning iteration\s+(\d+)/(\d+)"),
    "mean_reward": re.compile(r"Mean reward:\s+([-+0-9.eE]+)"),
    "episode_length": re.compile(r"Mean episode length:\s+([-+0-9.eE]+)"),
    "tracer_slide_reward": re.compile(r"Episode_Reward/tracer_slide_reward:\s+([-+0-9.eE]+)"),
    "error_vel_xy": re.compile(r"Metrics/base_velocity/error_vel_xy:\s+([-+0-9.eE]+)"),
    "error_vel_yaw": re.compile(r"Metrics/base_velocity/error_vel_yaw:\s+([-+0-9.eE]+)"),
    "time_out": re.compile(r"Episode_Termination/time_out:\s+([-+0-9.eE]+)"),
    "base_contact": re.compile(r"Episode_Termination/base_contact:\s+([-+0-9.eE]+)"),
    "training_time": re.compile(r"Training time:\s+([-+0-9.eE]+)\s+seconds"),
}


def parse_log(path: Path) -> dict[str, float | int | None]:
    rows: list[dict[str, float | int | None]] = []
    current: dict[str, float | int | None] = {}

    for line in path.read_text(errors="replace").splitlines():
        if m := PATTERNS["iteration"].search(line):
            if current:
                rows.append(current)
            current = {
                "iteration": int(m.group(1)),
                "max_iteration": int(m.group(2)),
            }
            continue

        for key, pattern in PATTERNS.items():
            if key == "iteration":
                continue
            m = pattern.search(line)
            if m:
                current[key] = float(m.group(1))

    if current:
        rows.append(current)

    if not rows:
        raise RuntimeError(f"No iteration rows parsed from {path}")

    final = rows[-1]
    if "training_time" not in final:
        # training time appears after the final iteration; parse globally as fallback
        text = path.read_text(errors="replace")
        if m := PATTERNS["training_time"].search(text):
            final["training_time"] = float(m.group(1))
        else:
            final["training_time"] = None

    return final


def fmt(v: float | int | None) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, int):
        return str(v)
    return f"{v:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracer-log", required=True)
    parser.add_argument("--baseline-log", required=True)
    args = parser.parse_args()

    tracer = parse_log(Path(args.tracer_log))
    baseline = parse_log(Path(args.baseline_log))

    keys = [
        "iteration",
        "mean_reward",
        "episode_length",
        "tracer_slide_reward",
        "error_vel_xy",
        "error_vel_yaw",
        "time_out",
        "base_contact",
        "training_time",
    ]

    print("| Metric | TRACER | Baseline | Delta(TRACER - Baseline) |")
    print("|---|---:|---:|---:|")
    for key in keys:
        tv = tracer.get(key)
        bv = baseline.get(key)

        if isinstance(tv, (int, float)) and isinstance(bv, (int, float)):
            delta = tv - bv
            delta_s = f"{delta:.4f}"
        else:
            delta_s = "N/A"

        print(f"| {key} | {fmt(tv)} | {fmt(bv)} | {delta_s} |")


if __name__ == "__main__":
    main()
