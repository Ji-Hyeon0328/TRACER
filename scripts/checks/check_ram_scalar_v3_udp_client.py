#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracer_core.highlevel.ram_scalar_v3_udp_client import RAMScalarV3UDPClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50230)
    ap.add_argument("--dataset", default="data/training/tracer_ram_scalar_window_dataset_v3.npz")
    ap.add_argument("--timeout-sec", type=float, default=0.5)
    args = ap.parse_args()

    d = np.load(args.dataset, allow_pickle=True)
    X = d["windows"]
    Y = d["labels"]
    label_names = [str(x) for x in d["label_names"]]

    client = RAMScalarV3UDPClient(
        host=args.host,
        port=args.port,
        timeout_sec=args.timeout_sec,
    )

    for idx in [0, 114]:
        out = client.query_features(X[idx], reset=True)
        target = {label_names[j]: float(Y[idx, j]) for j in range(len(label_names))}
        print(
            f"idx={idx:04d} ok={out.ok} "
            f"risk={out.risk:.6f} bad={out.bad_prob:.6f} good={out.good_prob:.6f} "
            f"target={target}"
        )
        print("raw_error:", out.raw.get("error"))


if __name__ == "__main__":
    main()
