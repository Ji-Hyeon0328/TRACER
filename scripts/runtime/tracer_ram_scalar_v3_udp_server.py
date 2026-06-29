#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracer_core.highlevel.ram_scalar_v3 import RAMScalarV3


def _json_default(x):
    try:
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    return str(x)


class RAMScalarV3UDPServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        model_path: str,
        max_datagram: int = 65535,
    ):
        self.host = host
        self.port = port
        self.max_datagram = max_datagram
        self.ram = RAMScalarV3(model_path)
        self.buffer = deque(maxlen=self.ram.window)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))

    def _reset(self):
        self.buffer.clear()

    def _handle(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if bool(msg.get("reset", False)):
            self._reset()

        # Supported payloads:
        # 1) {"row": {"mpc_vx": ..., ...}}
        # 2) {"rows": [{...}, {...}]}
        # 3) {"window": [[...22...], ...]}
        # 4) {"features": [[...22...], ...]}
        if "row" in msg:
            self.buffer.append(msg["row"])
            window = list(self.buffer)
        elif "rows" in msg:
            rows = msg["rows"]
            for r in rows:
                self.buffer.append(r)
            window = list(self.buffer)
        elif "window" in msg:
            # Numeric [T,D] payload.
            window = np.asarray(msg["window"], dtype=np.float32)
        elif "features" in msg:
            # Numeric [T,D] payload. This path is used by the smoke checker.
            window = np.asarray(msg["features"], dtype=np.float32)
        else:
            return {
                "ok": False,
                "error": "missing one of: row, rows, window, features",
                "expected_feature_names": self.ram.feature_names,
            }

        out = self.ram.predict_window(window)

        return {
            "ok": True,
            "stamp_wall": time.time(),
            "risk": out.risk,
            "bad_prob": out.bad_prob,
            "good_prob": out.good_prob,
            "probabilities": out.probabilities,
            "window_count": len(window) if hasattr(window, "__len__") else None,
            "model": str(self.ram.model_path),
            "label_names": out.label_names,
            "feature_names": out.feature_names,
        }

    def serve_forever(self):
        print(
            f"[TRACER] RAM scalar v3 UDP server listening on {self.host}:{self.port}",
            flush=True,
        )
        print(f"[TRACER] model={self.ram.model_path}", flush=True)
        print(f"[TRACER] window={self.ram.window} input_dim={self.ram.input_dim}", flush=True)

        while True:
            data, addr = self.sock.recvfrom(self.max_datagram)
            try:
                msg = json.loads(data.decode("utf-8"))
                resp = self._handle(msg)
            except Exception as exc:
                resp = {
                    "ok": False,
                    "error": repr(exc),
                    "stamp_wall": time.time(),
                }

            payload = json.dumps(resp, default=_json_default).encode("utf-8")
            self.sock.sendto(payload, addr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50230)
    ap.add_argument("--model", default="configs/learned_models/tracer_ram_scalar_v3_model.pt")
    args = ap.parse_args()

    server = RAMScalarV3UDPServer(
        host=args.host,
        port=args.port,
        model_path=args.model,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
