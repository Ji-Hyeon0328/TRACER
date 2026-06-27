#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path


def first_jsonl(path: str, terrain: str | None = None):
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if terrain is None or o.get("terrain") == terrain:
                return o
    raise RuntimeError(f"no sample found in {path} terrain={terrain}")


def query(host: str, port: int, payload: dict, timeout: float):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    sock.sendto(json.dumps(payload).encode("utf-8"), (host, port))
    data, _ = sock.recvfrom(65535)
    return json.loads(data.decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50420)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--terrain", default="flat_normal")
    ap.add_argument("--objective-data", default="data/objective_selector_dataset_v1/objective_selector_dataset_v1.jsonl")
    ap.add_argument("--ram-data", default="data/ram_window_dataset_v1/ram_windows_v1.jsonl")
    ap.add_argument("--gms-data", default="data/gms_dataset_v1/gms_dataset_v1.jsonl")
    args = ap.parse_args()

    obj = first_jsonl(args.objective_data, args.terrain)
    ram = first_jsonl(args.ram_data, args.terrain)
    gms = first_jsonl(args.gms_data, args.terrain)

    payload = {
        "request_id": f"smoke_{args.terrain}_{int(time.time())}",
        "terrain": args.terrain,
        "objective_x": obj["x"],
        "ram_x": ram["x"],
        "gms_x": gms["x"],
    }

    resp = query(args.host, args.port, payload, args.timeout)
    print(json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()
