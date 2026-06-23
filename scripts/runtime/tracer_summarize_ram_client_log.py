#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LOG = Path(os.environ.get("TRACER_RAM_CLIENT_LOG", "/tmp/tracer_online_ram_udp_client_v0.log"))
DEFAULT_CSV = ROOT / "data/sanity_results/tracer_ram_monitor_results_v2.csv"

KEYS = [
    "risk",
    "slip",
    "invalid",
    "valid",
    "fallen",
    "recovery",
    "sigma",
    "rho_norm",
    "score",
    "ctrl_risk",
    "ctrl_ema",
    "zmean",
    "z95",
    "zmax",
    "last_zmax",
]

PAT = re.compile(
    r"RAM "
    r"risk=(?P<risk>[-+0-9.eE]+) "
    r"slip=(?P<slip>[-+0-9.eE]+) "
    r"invalid=(?P<invalid>[-+0-9.eE]+) "
    r"valid=(?P<valid>[-+0-9.eE]+) "
    r"fallen=(?P<fallen>[-+0-9.eE]+) "
    r"recovery=(?P<recovery>[-+0-9.eE]+) "
    r"sigma=(?P<sigma>[-+0-9.eE]+) "
    r"rho_norm=(?P<rho_norm>[-+0-9.eE]+) "
    r"score=(?P<score>[-+0-9.eE]+) "
    r"ctrl_risk=(?P<ctrl_risk>[-+0-9.eE]+) "
    r"ctrl_ema=(?P<ctrl_ema>[-+0-9.eE]+) "
    r"zmean=(?P<zmean>[-+0-9.eE]+) "
    r"z95=(?P<z95>[-+0-9.eE]+) "
    r"zmax=(?P<zmax>[-+0-9.eE]+) "
    r"last_zmax=(?P<last_zmax>[-+0-9.eE]+)"
)


def parse_log(path: Path) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = PAT.search(line)
        if not m:
            continue
        rows.append({k: float(m.group(k)) for k in KEYS})
    return rows


def stats(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    vals = [r[key] for r in rows]
    tail3 = vals[-3:] if len(vals) >= 3 else vals
    tail5 = vals[-5:] if len(vals) >= 5 else vals

    return {
        f"{key}_first": vals[0],
        f"{key}_last": vals[-1],
        f"{key}_mean": mean(vals),
        f"{key}_tail3_mean": mean(tail3),
        f"{key}_tail5_mean": mean(tail5),
        f"{key}_max": max(vals),
        f"{key}_min": min(vals),
    }


def main() -> None:
    log_path = Path(os.environ.get("TRACER_RAM_CLIENT_LOG", str(DEFAULT_LOG))).expanduser()
    out_csv = Path(os.environ.get("TRACER_RAM_RESULT_CSV", str(DEFAULT_CSV))).expanduser()

    terrain_key = os.environ.get("TRACER_TERRAIN_KEY", "")
    world_name = os.environ.get("TRACER_WORLD_NAME", "")
    tag = os.environ.get("TRACER_SANITY_TAG", "")

    if not log_path.exists():
        raise FileNotFoundError(log_path)

    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"[TRACER] no RAM lines parsed from {log_path}")

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sanity_tag": tag,
        "terrain_key": terrain_key,
        "world_name": world_name,
        "log_path": str(log_path),
        "num_ram_samples": len(rows),
    }

    for k in KEYS:
        summary.update(stats(rows, k))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()

    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    print("[TRACER] RAM monitor summary")
    print("  log:", log_path)
    print("  csv:", out_csv)
    print("  tag:", tag)
    print("  terrain:", terrain_key)
    print("  world:", world_name)
    print("  samples:", len(rows))
    print(
        "  ctrl_ema:",
        f"first={summary['ctrl_ema_first']:.3f}",
        f"last={summary['ctrl_ema_last']:.3f}",
        f"mean={summary['ctrl_ema_mean']:.3f}",
        f"tail3={summary['ctrl_ema_tail3_mean']:.3f}",
        f"tail5={summary['ctrl_ema_tail5_mean']:.3f}",
        f"max={summary['ctrl_ema_max']:.3f}",
    )
    print(
        "  ctrl_risk:",
        f"mean={summary['ctrl_risk_mean']:.3f}",
        f"max={summary['ctrl_risk_max']:.3f}",
    )
    print(
        "  fallen/recovery:",
        f"fallen_mean={summary['fallen_mean']:.3f}",
        f"fallen_max={summary['fallen_max']:.3f}",
        f"recovery_mean={summary['recovery_mean']:.3f}",
        f"recovery_max={summary['recovery_max']:.3f}",
    )
    print(
        "  z:",
        f"zmean={summary['zmean_mean']:.3f}",
        f"z95={summary['z95_mean']:.3f}",
        f"zmax={summary['zmax_max']:.3f}",
    )


if __name__ == "__main__":
    main()
