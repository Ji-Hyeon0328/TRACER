#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MANIFEST_KEYS = [
    "tag",
    "npz_path",
    "summary_path",
    "world",
    "terrain_key",
    "policy_json",
    "style",
    "semantic_mode",
    "primitive_family",
    "command_vx",
    "command_yaw",
    "command_body_height",
    "command_clearance",
    "command_enable",
    "fall_like",
    "trajectory_cost_v0",
    "dx",
    "dy",
    "min_z",
    "max_abs_roll",
    "max_abs_pitch",
    "pose_source",
    "commit_hash",
    "source",
    "note",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    by_npz: dict[str, dict[str, Any]] = {}
    by_tag: dict[str, dict[str, Any]] = {}

    for row in rows:
        npz = str(row.get("npz_path", ""))
        tag = str(row.get("tag", ""))
        if npz:
            by_npz[npz] = row
        if tag:
            by_tag[tag] = row

    # Store both lookup modes in one dict with prefixed keys.
    out: dict[str, dict[str, Any]] = {}
    for k, v in by_npz.items():
        out[f"npz::{k}"] = v
    for k, v in by_tag.items():
        out[f"tag::{k}"] = v
    return out


def find_manifest(
    manifest: dict[str, dict[str, Any]],
    *,
    npz_path: str | None,
    tag: str | None,
) -> dict[str, Any]:
    if npz_path:
        hit = manifest.get(f"npz::{npz_path}")
        if hit is not None:
            return hit

    if tag:
        hit = manifest.get(f"tag::{tag}")
        if hit is not None:
            return hit

    return {}


def prefixed(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in MANIFEST_KEYS:
        out[f"{prefix}_{k}"] = row.get(k, "unknown")
    return out


def is_unknown(x: Any) -> bool:
    return str(x).strip().lower() in {"", "unknown", "none", "nan"}


def enrich_pair(pair: dict[str, Any], manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    winner_meta = find_manifest(
        manifest,
        npz_path=pair.get("winner_npz"),
        tag=pair.get("winner_tag"),
    )
    loser_meta = find_manifest(
        manifest,
        npz_path=pair.get("loser_npz"),
        tag=pair.get("loser_tag"),
    )

    out = dict(pair)
    out.update(prefixed("winner", winner_meta))
    out.update(prefixed("loser", loser_meta))

    out["winner_manifest_found"] = bool(winner_meta)
    out["loser_manifest_found"] = bool(loser_meta)

    out["same_world"] = (
        not is_unknown(out.get("winner_world"))
        and out.get("winner_world") == out.get("loser_world")
    )
    out["same_terrain_key"] = (
        not is_unknown(out.get("winner_terrain_key"))
        and out.get("winner_terrain_key") == out.get("loser_terrain_key")
    )
    out["same_policy_json"] = (
        not is_unknown(out.get("winner_policy_json"))
        and out.get("winner_policy_json") == out.get("loser_policy_json")
    )
    out["same_primitive_family"] = (
        not is_unknown(out.get("winner_primitive_family"))
        and out.get("winner_primitive_family") == out.get("loser_primitive_family")
    )

    out["known_world_pair"] = (
        not is_unknown(out.get("winner_world"))
        and not is_unknown(out.get("loser_world"))
    )
    out["known_style_pair"] = (
        not is_unknown(out.get("winner_style"))
        and not is_unknown(out.get("loser_style"))
    )

    out["pair_type"] = f"{out.get('winner_primitive_family', 'unknown')}__over__{out.get('loser_primitive_family', 'unknown')}"

    return out


def keep_pair(
    row: dict[str, Any],
    *,
    require_manifest: bool,
    require_known_world: bool,
    require_same_world: bool,
    drop_unknown_style: bool,
) -> bool:
    if require_manifest and not (row.get("winner_manifest_found") and row.get("loser_manifest_found")):
        return False
    if require_known_world and not row.get("known_world_pair"):
        return False
    if require_same_world and not row.get("same_world"):
        return False
    if drop_unknown_style and not row.get("known_style_pair"):
        return False
    return True


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        # Keep full nested winner/loser metric dictionaries in JSONL only.
        # CSV is intended to be a compact human-readable table.
        if k in {"winner", "loser"}:
            continue

        if isinstance(v, (dict, list)):
            out[k] = json.dumps(v, sort_keys=True)
        else:
            out[k] = v
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [flatten_for_csv(r) for r in rows]
    if not flat:
        path.write_text("")
        return

    priority = [
        "pair_id",
        "reason",
        "pair_type",
        "confidence",
        "winner_cost",
        "loser_cost",
        "cost_gap",
        "same_world",
        "same_policy_json",
        "same_primitive_family",
        "known_world_pair",
        "known_style_pair",
        "winner_tag",
        "winner_world",
        "winner_terrain_key",
        "winner_style",
        "winner_semantic_mode",
        "winner_primitive_family",
        "winner_fall_like",
        "loser_tag",
        "loser_world",
        "loser_terrain_key",
        "loser_style",
        "loser_semantic_mode",
        "loser_primitive_family",
        "loser_fall_like",
    ]

    all_fields = sorted({k for r in flat for k in r.keys()})
    fields = priority + [k for k in all_fields if k not in priority]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in flat:
            writer.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs-jsonl",
        default="data/preference_datasets/tracer_rollout_preference_pairs_v0.jsonl",
    )
    ap.add_argument(
        "--manifest-jsonl",
        default="data/rollout_manifests/tracer_rollout_manifest_v0.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        default="data/preference_datasets",
    )
    ap.add_argument("--require-manifest", action="store_true")
    ap.add_argument("--require-known-world", action="store_true")
    ap.add_argument("--require-same-world", action="store_true")
    ap.add_argument("--drop-unknown-style", action="store_true")
    args = ap.parse_args()

    pairs = load_jsonl(Path(args.pairs_jsonl))
    manifest = load_manifest(Path(args.manifest_jsonl))

    enriched = [enrich_pair(p, manifest) for p in pairs]
    filtered = [
        p for p in enriched
        if keep_pair(
            p,
            require_manifest=args.require_manifest,
            require_known_world=args.require_known_world,
            require_same_world=args.require_same_world,
            drop_unknown_style=args.drop_unknown_style,
        )
    ]

    out_dir = Path(args.out_dir)
    suffix = "filtered" if any([
        args.require_manifest,
        args.require_known_world,
        args.require_same_world,
        args.drop_unknown_style,
    ]) else "all"

    out_jsonl = out_dir / f"tracer_rollout_preference_pairs_with_manifest_v0_{suffix}.jsonl"
    out_csv = out_dir / f"tracer_rollout_preference_pairs_with_manifest_v0_{suffix}.csv"

    write_jsonl(out_jsonl, filtered)
    write_csv(out_csv, filtered)

    print(f"[TRACER] input pairs:       {len(pairs)}")
    print(f"[TRACER] enriched pairs:    {len(enriched)}")
    print(f"[TRACER] output pairs:      {len(filtered)}")
    print(f"[TRACER] wrote jsonl:       {out_jsonl}")
    print(f"[TRACER] wrote csv:         {out_csv}")

    known_world = sum(1 for p in enriched if p.get("known_world_pair"))
    same_world = sum(1 for p in enriched if p.get("same_world"))
    known_style = sum(1 for p in enriched if p.get("known_style_pair"))

    print()
    print(f"[TRACER] known_world pairs: {known_world}")
    print(f"[TRACER] same_world pairs:  {same_world}")
    print(f"[TRACER] known_style pairs: {known_style}")

    print("\n[TRACER] preview:")
    for p in filtered[:12]:
        print(
            f"{p.get('pair_id')} "
            f"{p.get('pair_type')} "
            f"conf={float(p.get('confidence', 0.0)):.2f} "
            f"gap={float(p.get('cost_gap', 0.0)):.3f} "
            f"same_world={p.get('same_world')} "
            f"winner={p.get('winner_style')} "
            f"loser={p.get('loser_style')}"
        )


if __name__ == "__main__":
    main()
