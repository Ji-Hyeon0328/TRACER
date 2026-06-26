#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = ROOT / "data/preference_datasets"
OUT_JSON = OUT_DIR / "tracer_negative_source_audit_v0.json"
OUT_CSV = OUT_DIR / "tracer_negative_source_audit_v0.csv"

PATTERNS = [
    "data/rollout_metrics/gms_sponge_downslope_hold_evidence_v0.json",
    "data/sanity_results/tracer_*slippery*.csv",
    "data/sanity_results/tracer_*backstep*.csv",
    "data/sanity_results/tracer_*micro*.csv",
    "data/sanity_results/tracer_*active_hold*.csv",
    "data/sanity_results/tracer_*primitive*.csv",
    "data/style_missions/style_mission_summary_fall.csv",
]

KEY_HINTS = [
    "terrain",
    "terrain_key",
    "world",
    "tag",
    "case",
    "style",
    "semantic",
    "decision",
    "valid",
    "invalid",
    "fallen",
    "fall",
    "recovery",
    "success",
    "dx",
    "dy",
    "min_rel_z",
    "min_z",
    "mpc_vx_mean",
    "gate",
    "risk",
    "score",
]


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return list(reader.fieldnames or []), rows
    except Exception:
        return [], []


def read_json_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return [], []

    if isinstance(data, list):
        rows = [x for x in data if isinstance(x, dict)]
    elif isinstance(data, dict):
        rows = [data]
    else:
        rows = []

    keys = sorted({k for r in rows for k in r.keys()})
    return keys, rows


def interesting_keys(keys: list[str]) -> list[str]:
    out = []
    lower = {k: k.lower() for k in keys}
    for k in keys:
        lk = lower[k]
        if any(h in lk for h in KEY_HINTS):
            out.append(k)
    return out


def sample_values(rows: list[dict[str, Any]], keys: list[str], n: int = 3) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k in keys:
        vals = []
        seen = set()
        for r in rows:
            v = str(r.get(k, ""))
            if v and v not in seen:
                vals.append(v)
                seen.add(v)
            if len(vals) >= n:
                break
        out[k] = vals
    return out


def infer_counts(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, int]:
    counts = {
        "rows_with_fall_like_true": 0,
        "rows_with_valid_like_false": 0,
        "rows_with_invalid_like_true": 0,
        "rows_with_success_like_false": 0,
    }

    fall_keys = [k for k in keys if "fall" in k.lower() or "fallen" in k.lower()]
    valid_keys = [k for k in keys if "valid" in k.lower()]
    invalid_keys = [k for k in keys if "invalid" in k.lower()]
    success_keys = [k for k in keys if "success" in k.lower()]

    true_vals = {"1", "true", "yes", "y"}
    false_vals = {"0", "false", "no", "n"}

    for r in rows:
        if any(str(r.get(k, "")).strip().lower() in true_vals for k in fall_keys):
            counts["rows_with_fall_like_true"] += 1
        if any(str(r.get(k, "")).strip().lower() in false_vals for k in valid_keys):
            counts["rows_with_valid_like_false"] += 1
        if any(str(r.get(k, "")).strip().lower() in true_vals for k in invalid_keys):
            counts["rows_with_invalid_like_true"] += 1
        if any(str(r.get(k, "")).strip().lower() in false_vals for k in success_keys):
            counts["rows_with_success_like_false"] += 1

    return counts


def main() -> None:
    paths: list[Path] = []
    for pat in PATTERNS:
        paths.extend(sorted(ROOT.glob(pat)))

    # de-duplicate while preserving order
    uniq = []
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)

    records = []
    for p in uniq:
        rel = p.relative_to(ROOT)
        if p.suffix.lower() == ".json":
            keys, rows = read_json_rows(p)
            ftype = "json"
        elif p.suffix.lower() == ".csv":
            keys, rows = read_csv_rows(p)
            ftype = "csv"
        else:
            continue

        ikeys = interesting_keys(keys)
        counts = infer_counts(rows, keys)
        rec = {
            "path": str(rel),
            "type": ftype,
            "num_rows": len(rows),
            "num_columns": len(keys),
            "interesting_keys": ikeys,
            "sample_values": sample_values(rows, ikeys),
            **counts,
        }
        records.append(rec)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(records, indent=2) + "\n")

    flat_fields = [
        "path",
        "type",
        "num_rows",
        "num_columns",
        "rows_with_fall_like_true",
        "rows_with_valid_like_false",
        "rows_with_invalid_like_true",
        "rows_with_success_like_false",
        "interesting_keys",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for r in records:
            rr = dict(r)
            rr["interesting_keys"] = "|".join(rr["interesting_keys"])
            writer.writerow({k: rr.get(k, "") for k in flat_fields})

    print("[TRACER] wrote", OUT_JSON)
    print("[TRACER] wrote", OUT_CSV)
    print(json.dumps({
        "num_sources": len(records),
        "outputs": {
            "json": str(OUT_JSON.relative_to(ROOT)),
            "csv": str(OUT_CSV.relative_to(ROOT)),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
