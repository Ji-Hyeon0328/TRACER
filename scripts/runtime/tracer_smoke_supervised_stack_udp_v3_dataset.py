#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path


def request(payload, host="127.0.0.1", port=50430, timeout=2.0):
    data = json.dumps(payload).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.sendto(data, (host, port))
        resp, _ = sock.recvfrom(65535)
        return json.loads(resp.decode("utf-8"))
    finally:
        sock.close()


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def objective_x_for_terrain(terrain: str):
    if terrain == "flat_normal":
        return [
            1.0, 0.0, 0.0,
            1.0, 0.5, 1.0,
            0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0,
            1.0,
            0.8,
        ]

    if terrain == "rough_mid":
        return [
            0.0, 1.0, 0.0,
            1.0, 0.40, 1.0,
            0.045, 0.075,
            0.75, 0.75, 0.75,
            1.0, 0.0,
            1.0,
            0.75,
        ]

    return [
        0.0, 0.0, 1.0,
        1.0, 0.40, 1.0,
        0.055, 0.080,
        0.75, 0.75, 0.75,
        1.0, 0.0,
        1.0,
        0.75,
    ]


def gms_x_for_terrain(terrain: str):
    if terrain == "flat_normal":
        return [
            0.28, 0.0, 0.295, 0.030, 1.0,
            0.55, 0.25, 0.20,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.1, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]

    if terrain == "rough_mid":
        return [
            0.040, 0.0, 0.350, 0.108, 1.0,
            0.30, 0.50, 0.20,
            1.5, 1.5, 0.75,
            1.0, 1.0, 0.0,
            0.52, 5.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ]

    return [
        0.038, 0.0, 0.353, 0.111, 1.0,
        0.25, 0.55, 0.20,
        1.5, 1.5, 0.75,
        1.0, 1.0, 0.0,
        0.52, 5.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ram_shadow_dataset_v2/ram_shadow_windows_v2.jsonl")
    ap.add_argument("--port", type=int, default=50430)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.data))

    for terrain in ["flat_normal", "rough_mid", "slope_5deg"]:
        matches = [r for r in rows if r.get("terrain") == terrain]
        if not matches:
            print(json.dumps({"terrain": terrain, "error": "no sample"}, indent=2))
            continue

        sample = matches[len(matches) // 2]

        payload = {
            "request_id": f"smoke_v3_dataset_{terrain}_{int(time.time())}",
            "objective_x": objective_x_for_terrain(terrain),
            "ram_x": sample["x"],
            "gms_x": gms_x_for_terrain(terrain),
        }

        resp = request(payload, port=args.port)

        compact = {
            "terrain": terrain,
            "ok": resp.get("ok"),
            "objective_beta": resp.get("objective", {}).get("beta"),
            "ram_ok": resp.get("ram", {}).get("ok"),
            "ram_intervention_score": resp.get("ram", {}).get("intervention_score"),
            "ram_future_override_mean": resp.get("ram", {}).get("future_override_mean"),
            "ram_labels": resp.get("ram", {}).get("labels"),
            "gms_label": resp.get("gms", {}).get("label"),
            "gms_prob": resp.get("gms", {}).get("prob"),
            "target_y": sample["y"],
            "label_names": sample["label_names"],
        }

        print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
