#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


JSONL_FILES = [
    "phase_b_rollout_index_v1.jsonl",
    "phase_b_objective_preference_pairs_v1.jsonl",
    "phase_b_ram_teacher_student_episode_labels_v1.jsonl",
    "phase_b_discrete_theta_lite_ppo_rows_v1.jsonl",
]


def read_jsonl(p):
    rows = []
    if not p.exists():
        return rows
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def write_csv_from_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (dict, list)):
                continue
            keys.add(k)
    keys = sorted(keys)

    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def row_key(r):
    for keys in [
        ("result_path",),
        ("source_path",),
        ("run_dir",),
        ("world_name", "action_name", "episode_index"),
        ("world", "action", "episode_index"),
    ]:
        vals = [r.get(k) for k in keys]
        if all(v is not None for v in vals):
            return tuple(vals)
    return json.dumps(r, sort_keys=True)


def infer_world_action(r):
    w = r.get("world_name") or r.get("world") or ""
    a = r.get("action_name") or r.get("action") or r.get("selected_action_name") or ""
    return w, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        nargs="+",
        default=[
            "data/phase_b_training_pipeline_v5_merged_balanced",
            "data/phase_b_anchor_mixture_v8_eval",
            "data/phase_b_anchor_mixture_v8b_sweep_eval",
            "data/phase_b_anchor_mixture_v8c_phase_scheduled_eval",
        ],
    )
    ap.add_argument("--out-dir", default="data/phase_b_training_pipeline_v6_anchor_mixture_merged")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "phase_b_training_pipeline_v6_anchor_mixture_merged_report",
        "sources": args.sources,
        "out_dir": args.out_dir,
        "files": {},
    }

    for fname in JSONL_FILES:
        merged = []
        seen = set()

        for src in args.sources:
            src_path = Path(src)
            rows = read_jsonl(src_path / fname)
            for r in rows:
                rr = dict(r)
                rr.setdefault("source_dataset", str(src_path))

                # Rollout-like files should be deduplicated.
                # Preference pairs can safely keep duplicate-looking rows because
                # each source dataset creates its own pair set.
                if fname == "phase_b_rollout_index_v1.jsonl":
                    k = row_key(rr)
                    if k in seen:
                        continue
                    seen.add(k)

                merged.append(rr)

        write_jsonl(out_dir / fname, merged)
        report["files"][fname] = {
            "rows": len(merged),
        }

        if fname == "phase_b_rollout_index_v1.jsonl":
            write_csv_from_jsonl(out_dir / "phase_b_rollout_index_v1.csv", merged)

            by_action = defaultdict(lambda: {"n": 0})
            for r in merged:
                w, a = infer_world_action(r)
                by_action[f"{w}::{a}"]["n"] += 1

            report["rollout_summary_by_world_action"] = dict(sorted(by_action.items()))

    # Copy/compose markdown report.
    lines = []
    lines.append("# Phase-B v6 Anchor-Mixture Merged Dataset")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    for s in args.sources:
        lines.append(f"- `{s}`")

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| file | rows |")
    lines.append("|---|---:|")
    for fname, info in report["files"].items():
        lines.append(f"| `{fname}` | {info['rows']} |")

    lines.append("")
    lines.append("## Rollouts by world::action")
    lines.append("")
    lines.append("| world::action | n |")
    lines.append("|---|---:|")
    for k, v in report.get("rollout_summary_by_world_action", {}).items():
        lines.append(f"| {k} | {v['n']} |")

    md = "\n".join(lines) + "\n"

    with open(out_dir / "phase_b_training_pipeline_v6_anchor_mixture_merged_report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    with open(out_dir / "phase_b_training_pipeline_v6_anchor_mixture_merged_report.md", "w") as f:
        f.write(md)

    print(md)
    print("[wrote]", out_dir)


if __name__ == "__main__":
    main()
