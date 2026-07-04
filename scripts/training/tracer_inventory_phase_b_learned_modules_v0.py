#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


KEYWORDS = [
    "objective",
    "selector",
    "irl",
    "preference",
    "reward_model",
    "beta",
    "ram",
    "mismatch",
    "rho",
    "sigma",
    "uncertainty",
    "risk"
]

SEARCH_DIRS = [
    "configs",
    "scripts",
    "tracer_core",
    "artifacts",
    "reports",
    "data"
]

CHECKPOINT_SUFFIXES = [
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".joblib"
]

TEXT_SUFFIXES = [
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".csv",
    ".jsonl"
]


def looks_relevant(path):
    s = str(path).lower()
    return any(k in s for k in KEYWORDS)


def classify(path):
    name = str(path).lower()
    if "ram" in name or "mismatch" in name or "rho" in name or "sigma" in name or "uncertainty" in name:
        return "ram"
    if "objective" in name or "selector" in name or "irl" in name or "preference" in name or "beta" in name or "reward_model" in name:
        return "objective_selector"
    return "unknown"


def safe_head(path, max_chars=1600):
    try:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            return ""
        txt = path.read_text(errors="ignore")
        return txt[:max_chars]
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = []

    for d in SEARCH_DIRS:
        root = Path(d)
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            if not looks_relevant(p):
                continue

            suffix = p.suffix.lower()
            kind = classify(p)
            file_type = "checkpoint" if suffix in CHECKPOINT_SUFFIXES else "text_or_config"

            rows.append({
                "path": str(p),
                "kind": kind,
                "file_type": file_type,
                "suffix": suffix,
                "size_bytes": p.stat().st_size,
                "head": safe_head(p)
            })

    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)

    report = {
        "schema": "phase_b_learned_module_inventory_v0",
        "purpose": "Find previously built or trained Objective Selector / RAM artifacts and determine what can be connected to the current Phase-B runner.",
        "search_dirs": SEARCH_DIRS,
        "keywords": KEYWORDS,
        "num_files": len(rows),
        "by_kind_counts": {k: len(v) for k, v in by_kind.items()},
        "files": rows
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Learned Module Inventory")
    lines.append("")
    lines.append("This report inventories existing Objective Selector / RAM / IRL / preference / uncertainty artifacts in the repository.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| kind | count |")
    lines.append("|---|---:|")
    for k, v in sorted(report["by_kind_counts"].items()):
        lines.append(f"| {k} | {v} |")

    lines.append("")
    lines.append("## Candidate Files")
    lines.append("")
    lines.append("| kind | type | path | size |")
    lines.append("|---|---|---|---:|")
    for r in sorted(rows, key=lambda x: (x["kind"], x["path"])):
        lines.append(f"| {r['kind']} | {r['file_type']} | `{r['path']}` | {r['size_bytes']} |")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "num_files": report["num_files"],
        "by_kind_counts": report["by_kind_counts"]
    }, indent=2, sort_keys=True))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
