#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

fusion_csv = ROOT / "data/sanity_results/tracer_fusion_sanity_results.csv"
ram_csv = ROOT / "data/sanity_results/tracer_ram_monitor_results_tailcheck.csv"
out_csv = ROOT / "data/sanity_results/tracer_fusion_ram_joined_results.csv"

if not fusion_csv.exists():
    raise FileNotFoundError(fusion_csv)
if not ram_csv.exists():
    raise FileNotFoundError(ram_csv)

fusion = pd.read_csv(fusion_csv)
ram = pd.read_csv(ram_csv)

# Keep latest row per tag/terrain in case multiple repeated runs exist.
if "sanity_tag" in fusion.columns:
    fusion = fusion.drop_duplicates(subset=["sanity_tag"], keep="last")
else:
    raise SystemExit("[TRACER] fusion CSV has no sanity_tag column")

ram = ram.drop_duplicates(subset=["sanity_tag"], keep="last")

joined = pd.merge(
    fusion,
    ram,
    on="sanity_tag",
    how="inner",
    suffixes=("_fusion", "_ram"),
)

# Prefer compact table columns when available.
preferred = [
    "sanity_tag",
    "terrain_key_fusion",
    "world_name_fusion",
    "fused_mode",
    "semantic_mode",
    "style",
    "decision",
    "valid_locomotion_candidate",
    "stable_hold_candidate",
    "fallen_relative_z_based",
    "delta_x",
    "delta_y",
    "min_rel_z",
    "ctrl_ema_mean",
    "ctrl_ema_tail3_mean",
    "ctrl_ema_tail5_mean",
    "ctrl_ema_last",
    "ctrl_ema_max",
    "ctrl_risk_mean",
    "fallen_mean",
    "fallen_tail3_mean",
    "fallen_tail5_mean",
    "fallen_max",
    "recovery_mean",
    "recovery_tail5_mean",
    "sigma_mean",
    "zmean_mean",
    "z95_mean",
    "zmax_max",
]

cols = [c for c in preferred if c in joined.columns]
compact = joined[cols].copy()

out_csv.parent.mkdir(parents=True, exist_ok=True)
compact.to_csv(out_csv, index=False)

print("[TRACER] joined fusion/RAM results")
print("  fusion:", fusion_csv)
print("  ram:   ", ram_csv)
print("  out:   ", out_csv)
print()
print(compact.tail(20).to_string(index=False))
