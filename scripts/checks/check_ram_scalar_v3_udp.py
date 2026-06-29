#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50230)
    ap.add_argument("--dataset", default="data/training/tracer_ram_scalar_window_dataset_v3.npz")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=2.0)
    args = ap.parse_args()

    d = np.load(args.dataset, allow_pickle=True)
    X = d["windows"]
    Y = d["labels"]
    label_names = [str(x) for x in d["label_names"]]

    idx = max(0, min(int(args.idx), len(X) - 1))
    target = {label_names[j]: float(Y[idx, j]) for j in range(len(label_names))}

    msg = {
        "reset": True,
        "features": X[idx].astype(float).tolist(),
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(args.timeout)

    sock.sendto(json.dumps(msg).encode("utf-8"), (args.host, args.port))
    data, _ = sock.recvfrom(65535)
    resp = json.loads(data.decode("utf-8"))

    print("[TRACER] RAM scalar v3 UDP smoke")
    print("target:", target)
    print("response:", json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()
