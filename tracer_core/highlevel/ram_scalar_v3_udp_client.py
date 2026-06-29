from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np


@dataclass
class RAMScalarV3UDPOutput:
    ok: bool
    risk: float
    bad_prob: float
    good_prob: float
    probabilities: Dict[str, float]
    age_sec: float
    raw: Dict[str, Any]


class RAMScalarV3UDPClient:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 50230,
        timeout_sec: float = 0.02,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(self.timeout_sec)

        self.last_output: RAMScalarV3UDPOutput | None = None

    def _request(self, msg: Dict[str, Any]) -> RAMScalarV3UDPOutput:
        now = time.time()
        try:
            payload = json.dumps(msg).encode("utf-8")
            self.sock.sendto(payload, (self.host, self.port))
            data, _ = self.sock.recvfrom(65535)
            resp = json.loads(data.decode("utf-8"))

            stamp = float(resp.get("stamp_wall", now))
            probs = resp.get("probabilities", {}) or {}

            out = RAMScalarV3UDPOutput(
                ok=bool(resp.get("ok", False)),
                risk=float(resp.get("risk", 1.0)),
                bad_prob=float(resp.get("bad_prob", probs.get("future_bad_locomotion", 1.0))),
                good_prob=float(resp.get("good_prob", probs.get("future_good_locomotion", 0.0))),
                probabilities={str(k): float(v) for k, v in probs.items()},
                age_sec=max(0.0, time.time() - stamp),
                raw=resp,
            )
            self.last_output = out
            return out

        except Exception as exc:
            # Fail-safe: if RAM server is unavailable, return high risk.
            return RAMScalarV3UDPOutput(
                ok=False,
                risk=1.0,
                bad_prob=1.0,
                good_prob=0.0,
                probabilities={},
                age_sec=float("inf"),
                raw={"ok": False, "error": repr(exc)},
            )

    def reset(self) -> RAMScalarV3UDPOutput:
        return self._request({"reset": True, "features": []})

    def query_row(self, row: Mapping[str, float], *, reset: bool = False) -> RAMScalarV3UDPOutput:
        return self._request({
            "reset": bool(reset),
            "row": dict(row),
        })

    def query_rows(
        self,
        rows: Sequence[Mapping[str, float]],
        *,
        reset: bool = False,
    ) -> RAMScalarV3UDPOutput:
        return self._request({
            "reset": bool(reset),
            "rows": [dict(r) for r in rows],
        })

    def query_features(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
        *,
        reset: bool = False,
    ) -> RAMScalarV3UDPOutput:
        arr = np.asarray(features, dtype=float)
        return self._request({
            "reset": bool(reset),
            "features": arr.tolist(),
        })
