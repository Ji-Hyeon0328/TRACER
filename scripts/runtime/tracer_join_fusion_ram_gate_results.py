#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

fusion_csv = Path(os.environ.get(
    "TRACER_FUSION_SANITY_CSV",
    str(ROOT / "data/sanity_results/tracer_fusion_sanity_results.csv"),
)).expanduser()

ram_csv = Path(os.environ.get(
    "TRACER_RAM_MONITOR_CSV",
    str(ROOT / "data/sanity_results/tracer_ram_monitor_results.csv"),
)).expanduser()

gate_csv = Path(os.environ.get(
    "TRACER_RAM_GATE_CSV",
    str(ROOT / "data/sanity_results/tracer_ram_gate_monitor_results.csv"),
)).expanduser()

out_csv = Path(os.environ.get(
    "TRACER_JOINED_OUT_CSV",
    str(ROOT / "data/sanity_results/tracer_fusion_ram_gate_joined_results.csv"),
)).expanduser()

for p in [fusion_csv, ram_csv, gate_csv]:
    if not p.exists():
        raise FileNotFoundError(p)

fusion = pd.read_csv(fusion_csv)
ram = pd.read_csv(ram_csv)
gate = pd.read_csv(gate_csv)

if "sanity_tag" not in fusion.columns:
    raise SystemExit("[TRACER] fusion CSV has no sanity_tag")
if "sanity_tag" not in ram.columns:
    raise SystemExit("[TRACER] RAM CSV has no sanity_tag")
if "sanity_tag" not in gate.columns:
    raise SystemExit("[TRACER] gate CSV has no sanity_tag")

fusion = fusion.drop_duplicates(subset=["sanity_tag"], keep="last")
ram = ram.drop_duplicates(subset=["sanity_tag"], keep="last")
gate = gate.drop_duplicates(subset=["sanity_tag"], keep="last")

joined = fusion.merge(
    ram,
    on="sanity_tag",
    how="inner",
    suffixes=("_fusion", "_ram"),
)

joined = joined.merge(
    gate,
    on="sanity_tag",
    how="inner",
    suffixes=("", "_gate"),
)

preferred = [
    "sanity_tag",

    # V1 / mission result
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

    # RAM monitor
    "ctrl_ema_mean",
    "ctrl_ema_tail3_mean",
    "ctrl_ema_tail5_mean",
    "ctrl_ema_last",
    "ctrl_ema_max",
    "fallen_mean",
    "fallen_tail3_mean",
    "fallen_tail5_mean",
    "sigma_mean",
    "sigma_tail5_mean",
    "zmean_mean",
    "zmax_max",

    # RAM gate
    "num_gate_samples",
    "v1_mode_last",
    "semantic_last",
    "ram_level_last",
    "action_last",
    "stable_frac",
    "caution_frac",
    "unstable_frac",
    "would_override_mean",
    "ctrl_ema_tail5_mean_gate",
    "fallen_tail5_mean_gate",
    "sigma_tail5_mean_gate",
]

cols = [c for c in preferred if c in joined.columns]
compact = joined[cols].copy()

out_csv.parent.mkdir(parents=True, exist_ok=True)
compact.to_csv(out_csv, index=False)

print("[TRACER] joined fusion + RAM + gate results")
print("  fusion:", fusion_csv)
print("  ram:   ", ram_csv)
print("  gate:  ", gate_csv)
print("  out:   ", out_csv)
print()
print(compact.tail(20).to_string(index=False))
