#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


PROPRIO_DIM = 57


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def b(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


def parse_vector_cell(s: str) -> List[float]:
    """Parse a vector cell from recorder CSV.

    The recorder has historically used several formats:
      - JSON-ish list: "[1, 2, 3]"
      - whitespace-separated list
      - semicolon-separated list
    This parser is intentionally permissive.
    """
    if s is None:
        return []
    txt = str(s).strip()
    if not txt:
        return []

    # Remove simple wrappers.
    txt = txt.replace("[", " ").replace("]", " ")
    txt = txt.replace("array('d',", " ").replace(")", " ")
    txt = txt.replace(",", " ").replace(";", " ")

    vals = []
    for tok in txt.split():
        try:
            vals.append(float(tok))
        except Exception:
            pass
    return vals


def find_col(fieldnames: List[str], candidates: List[str]) -> str | None:
    lower = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in fieldnames:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def read_episode_csv(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    with path.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
        fields = rows[0].keys() if rows else []

    fields = list(fields)

    proprio_col = find_col(fields, [
        "proprio_vector",
        "proprio",
        "proprio_vec",
        "tracer_proprio_vector",
    ])

    if proprio_col is None:
        raise RuntimeError(f"Could not find proprio vector column in {path}; fields={fields}")

    scalar_candidates = {
        "ram_run_fallen": ["ram_run_fallen", "run_fallen", "pred_run_fallen"],
        "ram_recovery_needed": ["ram_recovery_needed", "recovery_needed", "pred_recovery_needed"],
        "mpc_enable": ["mpc_enable", "enable"],
        "mpc_vx": ["mpc_vx", "vx", "target_vx"],
        "debug_fresh": ["debug_fresh", "fresh", "is_fresh"],
    }
    scalar_cols = {k: find_col(fields, v) for k, v in scalar_candidates.items()}

    X = []
    scalars = {k: [] for k in scalar_candidates}

    for r in rows:
        vec = parse_vector_cell(r.get(proprio_col, ""))
        if len(vec) < PROPRIO_DIM:
            vec = vec + [0.0] * (PROPRIO_DIM - len(vec))
        elif len(vec) > PROPRIO_DIM:
            vec = vec[:PROPRIO_DIM]
        X.append(vec)

        for k, col in scalar_cols.items():
            scalars[k].append(f(r.get(col, 0.0)) if col else 0.0)

    return np.asarray(X, dtype=np.float32), {k: np.asarray(v, dtype=np.float32) for k, v in scalars.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-rows-csv", default="reports/tracer_ram_gms_data_v2_core_hard_valid_rows.csv")
    ap.add_argument("--out-npz", default="data/training/tracer_ram_window_dataset_v2.npz")
    ap.add_argument("--summary-json", default="reports/tracer_ram_window_dataset_v2_summary.json")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--future", type=int, default=10)
    args = ap.parse_args()

    with Path(args.valid_rows_csv).open(newline="") as fp:
        valid_rows = list(csv.DictReader(fp))

    windows = []
    labels = []
    meta = []

    skipped = []

    for rr in valid_rows:
        summary_path = Path(rr["summary_json"])
        if not summary_path.exists():
            skipped.append({"summary": str(summary_path), "reason": "summary_missing"})
            continue

        summary = json.loads(summary_path.read_text())
        step_csv = Path(summary.get("step_csv", ""))
        if not step_csv.exists():
            skipped.append({"summary": str(summary_path), "reason": "step_csv_missing", "step_csv": str(step_csv)})
            continue

        try:
            X, S = read_episode_csv(step_csv)
        except Exception as exc:
            skipped.append({"summary": str(summary_path), "reason": f"read_failed:{exc}"})
            continue

        T = len(X)
        if T < args.window:
            skipped.append({"summary": str(summary_path), "reason": f"too_short:T={T}"})
            continue

        episode_success = b(rr.get("success_proxy"))
        episode_fallen_p90 = f(rr.get("ram_run_fallen_p90"))
        episode_recovery_mean = f(rr.get("ram_recovery_needed_mean"))
        episode_fresh = f(rr.get("debug_fresh_rate_1p0s"))
        episode_invalid = episode_fresh < 0.90 or f(rr.get("mpc_vx_mean")) <= 0.010
        episode_unstable = episode_fallen_p90 >= 0.70 or episode_recovery_mean >= 0.20
        episode_bad = episode_invalid or episode_unstable or (not episode_success)

        for start in range(0, T - args.window + 1, args.stride):
            end = start + args.window
            fut_end = min(T, end + args.future)

            future_fallen = float(np.max(S["ram_run_fallen"][end:fut_end])) if fut_end > end else episode_fallen_p90
            future_recovery = float(np.max(S["ram_recovery_needed"][end:fut_end])) if fut_end > end else episode_recovery_mean

            # Combine dense online RAM labels with episode-level labels.
            y_fallen = float(max(future_fallen, 1.0 if episode_fallen_p90 >= 0.70 else 0.0))
            y_recovery = float(max(future_recovery, 1.0 if episode_recovery_mean >= 0.20 else 0.0))
            y_invalid = float(1.0 if episode_invalid else 0.0)
            y_bad = float(1.0 if episode_bad else 0.0)
            y_good = float(1.0 if (episode_success and not episode_bad) else 0.0)

            windows.append(X[start:end])
            labels.append([y_fallen, y_recovery, y_invalid, y_bad, y_good])
            meta.append({
                "terrain": rr.get("terrain", ""),
                "style": rr.get("forced_style", ""),
                "summary_json": str(summary_path),
                "step_csv": str(step_csv),
                "start": start,
                "end": end,
                "episode_success": episode_success,
                "episode_fallen_p90": episode_fallen_p90,
                "episode_recovery_mean": episode_recovery_mean,
                "episode_fresh1": episode_fresh,
            })

    if not windows:
        raise SystemExit("No RAM windows built")

    Xw = np.asarray(windows, dtype=np.float32)
    Y = np.asarray(labels, dtype=np.float32)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        windows=Xw,
        labels=Y,
        label_names=np.asarray([
            "future_fallen",
            "future_recovery_needed",
            "future_invalid_runtime",
            "future_bad_locomotion",
            "future_good_locomotion",
        ]),
        meta_json=np.asarray([json.dumps(m) for m in meta]),
    )

    summary = {
        "valid_rows_csv": args.valid_rows_csv,
        "out_npz": str(out_npz),
        "window": args.window,
        "stride": args.stride,
        "future": args.future,
        "n_episodes": len(valid_rows),
        "n_windows": int(Xw.shape[0]),
        "window_shape": list(Xw.shape),
        "label_shape": list(Y.shape),
        "label_means": {
            "future_fallen": float(Y[:, 0].mean()),
            "future_recovery_needed": float(Y[:, 1].mean()),
            "future_invalid_runtime": float(Y[:, 2].mean()),
            "future_bad_locomotion": float(Y[:, 3].mean()),
            "future_good_locomotion": float(Y[:, 4].mean()),
        },
        "skipped": skipped,
    }

    Path(args.summary_json).write_text(json.dumps(summary, indent=2))

    print("[TRACER] RAM window dataset v2")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
