#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracer_core.highlevel.ram_scalar_v3 import RAMScalarV3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/learned_models/tracer_ram_scalar_v3_model.pt")
    ap.add_argument("--dataset", default="data/training/tracer_ram_scalar_window_dataset_v3.npz")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    ram = RAMScalarV3(args.model)

    d = np.load(args.dataset, allow_pickle=True)
    X = d["windows"]
    Y = d["labels"]
    label_names = [str(x) for x in d["label_names"]]

    print("[TRACER] RAM scalar v3 smoke")
    print("model:", args.model)
    print("dataset:", args.dataset)
    print("windows:", X.shape)
    print("labels:", Y.shape)
    print("label_names:", label_names)

    # Show a few diverse samples: first, middle, last, and then evenly spaced.
    idxs = sorted(set(
        [0, len(X)//2, len(X)-1]
        + [int(i) for i in np.linspace(0, len(X)-1, args.n)]
    ))

    for idx in idxs[: args.n]:
        out = ram.predict_window(X[idx])
        target = {label_names[j]: float(Y[idx, j]) for j in range(len(label_names))}
        print(
            f"idx={idx:04d} "
            f"risk={out.risk:.3f} "
            f"bad={out.bad_prob:.3f} "
            f"good={out.good_prob:.3f} "
            f"target={target} "
            f"pred={out.probabilities}"
        )


if __name__ == "__main__":
    main()
