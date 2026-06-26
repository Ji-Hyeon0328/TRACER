#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LOG = Path(os.environ.get("TRACER_RAM_GATE_MONITOR_LOG", "/tmp/tracer_ram_gate_monitor.log"))
DEFAULT_CSV = ROOT / "data/sanity_results/tracer_ram_gate_monitor_results.csv"

PAT = re.compile(
    r"terrain=(?P<terrain>\S+) "
    r"v1_mode=(?P<v1_mode>\S+) "
    r"semantic=(?P<semantic>\S+) "
    r"style=(?P<style>\S+) "
    r"ram_level=(?P<ram_level>\S+) "
    r"action=(?P<action>\S+) "
    r"would_override=(?P<would_override>[-+0-9.eE]+) "
    r"ctrl_ema=(?P<ctrl_ema>[-+0-9.eE]+) "
    r"ctrl_risk=(?P<ctrl_risk>[-+0-9.eE]+) "
    r"fallen=(?P<fallen>[-+0-9.eE]+) "
    r"recovery=(?P<recovery>[-+0-9.eE]+) "
    r"sigma=(?P<sigma>[-+0-9.eE]+) "
    r"vx_scale=(?P<vx_scale>[-+0-9.eE]+) "
    r"h_delta=(?P<h_delta>[-+0-9.eE]+) "
    r"clr_delta=(?P<clr_delta>[-+0-9.eE]+)"
)

NUM_KEYS = [
    "would_override",
    "ctrl_ema",
    "ctrl_risk",
    "fallen",
    "recovery",
    "sigma",
    "vx_scale",
    "h_delta",
    "clr_delta",
]


def parse_log(path: Path) -> list[dict]:
    rows = []

    required = {
        "terrain",
        "v1_mode",
        "semantic",
        "style",
        "ram_level",
        "action",
        "would_override",
        "ctrl_ema",
        "ctrl_risk",
        "fallen",
        "recovery",
        "sigma",
        "vx_scale",
        "h_delta",
        "clr_delta",
    }

    optional_defaults = {
        "raw_ram_level": "",
        "calibration": "",
    }

    for line in path.read_text(errors="replace").splitlines():
        if "terrain=" not in line or "ram_level=" not in line or "action=" not in line:
            continue

        kv = {}
        for tok in line.split():
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            if not k:
                continue
            kv[k] = v

        if not required.issubset(kv):
            continue

        row = {k: kv[k] for k in required}
        for k, v in optional_defaults.items():
            row[k] = kv.get(k, v)

        for k in NUM_KEYS:
            row[k] = float(row[k])

        rows.append(row)

    return rows


def tail_mean(vals: list[float], n: int) -> float:
    if not vals:
        return 0.0
    return mean(vals[-n:] if len(vals) >= n else vals)


def frac(rows: list[dict], key: str, value: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r[key] == value) / len(rows)


def summarize(rows: list[dict], log_path: Path) -> dict:
    if not rows:
        raise SystemExit(f"[TRACER] no gate advice lines parsed from {log_path}")

    first = rows[0]
    last = rows[-1]

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sanity_tag": os.environ.get("TRACER_SANITY_TAG", ""),
        "terrain_key": os.environ.get("TRACER_TERRAIN_KEY", first["terrain"]),
        "world_name": os.environ.get("TRACER_WORLD_NAME", ""),
        "log_path": str(log_path),
        "num_gate_samples": len(rows),
        "v1_mode_first": first["v1_mode"],
        "v1_mode_last": last["v1_mode"],
        "semantic_first": first["semantic"],
        "semantic_last": last["semantic"],
        "style_first": first["style"],
        "style_last": last["style"],
        "ram_level_first": first["ram_level"],
        "ram_level_last": last["ram_level"],
        "raw_ram_level_first": first.get("raw_ram_level", ""),
        "raw_ram_level_last": last.get("raw_ram_level", ""),
        "action_first": first["action"],
        "action_last": last["action"],
        "calibration_first": first.get("calibration", ""),
        "calibration_last": last.get("calibration", ""),
        "stable_frac": frac(rows, "ram_level", "stable"),
        "raw_stable_frac": frac(rows, "raw_ram_level", "stable"),
        "raw_caution_frac": frac(rows, "raw_ram_level", "caution"),
        "raw_unstable_frac": frac(rows, "raw_ram_level", "unstable"),
        "caution_frac": frac(rows, "ram_level", "caution"),
        "unstable_frac": frac(rows, "ram_level", "unstable"),
    }

    for k in NUM_KEYS:
        vals = [float(r[k]) for r in rows]
        out[f"{k}_first"] = vals[0]
        out[f"{k}_last"] = vals[-1]
        out[f"{k}_mean"] = mean(vals)
        out[f"{k}_tail3_mean"] = tail_mean(vals, 3)
        out[f"{k}_tail5_mean"] = tail_mean(vals, 5)
        out[f"{k}_max"] = max(vals)
        out[f"{k}_min"] = min(vals)

    return out


def main() -> None:
    log_path = Path(os.environ.get("TRACER_RAM_GATE_MONITOR_LOG", str(DEFAULT_LOG))).expanduser()
    out_csv = Path(os.environ.get("TRACER_RAM_GATE_RESULT_CSV", str(DEFAULT_CSV))).expanduser()

    rows = parse_log(log_path)
    summary = summarize(rows, log_path)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()

    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    print("[TRACER] RAM gate monitor summary")
    print("  log:", log_path)
    print("  csv:", out_csv)
    print("  tag:", summary["sanity_tag"])
    print("  terrain:", summary["terrain_key"])
    print("  samples:", summary["num_gate_samples"])
    print(
        "  levels:",
        f"stable={summary['stable_frac']:.3f}",
        f"caution={summary['caution_frac']:.3f}",
        f"unstable={summary['unstable_frac']:.3f}",
    )
    print(
        "  raw levels:",
        f"stable={summary.get('raw_stable_frac', 0.0):.3f}",
        f"caution={summary.get('raw_caution_frac', 0.0):.3f}",
        f"unstable={summary.get('raw_unstable_frac', 0.0):.3f}",
    )
    print(
        "  action:",
        f"first={summary['action_first']}",
        f"last={summary['action_last']}",
        f"would_override_frac={summary['would_override_mean']:.3f}",
    )
    print(
        "  calibration:",
        f"first={summary.get('calibration_first', '')}",
        f"last={summary.get('calibration_last', '')}",
    )
    print(
        "  ctrl_ema:",
        f"mean={summary['ctrl_ema_mean']:.3f}",
        f"tail5={summary['ctrl_ema_tail5_mean']:.3f}",
        f"max={summary['ctrl_ema_max']:.3f}",
    )
    print(
        "  fallen/sigma:",
        f"fallen_tail5={summary['fallen_tail5_mean']:.3f}",
        f"sigma_tail5={summary['sigma_tail5_mean']:.3f}",
    )


if __name__ == "__main__":
    main()
