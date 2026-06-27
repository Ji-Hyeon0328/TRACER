#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn


class ObjectiveSelectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class RamInterventionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GMSMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def tensor_norm(x, mean, std, device):
    xt = torch.tensor(x, dtype=torch.float32, device=device).view(1, -1)
    mean = mean.to(device).view(1, -1)
    std = std.to(device).view(1, -1).clamp_min(1e-6)
    return (xt - mean) / std



class _ObjectiveSelectorMLPv2(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, output_dim: int = 3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def load_objective(path, device):
    b = torch.load(path, map_location=device)

    # Objective Selector v2 checkpoint format.
    # Produced by scripts/training/tracer_train_objective_selector_v2.py
    if b.get("model_type", "") == "ObjectiveSelectorMLPv2" or "state_dict" in b:
        input_dim = int(b.get("input_dim", 15))
        output_dim = int(b.get("output_dim", 3))
        hidden = int(b.get("hidden", 64))

        m = _ObjectiveSelectorMLPv2(
            input_dim=input_dim,
            hidden=hidden,
            output_dim=output_dim,
        ).to(device)
        m.load_state_dict(b["state_dict"])
        m.eval()

        return {
            "model": m,
            "bundle": b,
            "beta_names": b.get("beta_names", ["motion", "stability", "energy"]),
            "x_mean": torch.tensor(
                b.get("x_mean", [0.0] * input_dim),
                dtype=torch.float32,
                device=device,
            ),
            "x_std": torch.tensor(
                b.get("x_std", [1.0] * input_dim),
                dtype=torch.float32,
                device=device,
            ).clamp_min(1e-6),
            "feature_names": b.get("feature_names", []),
            "target_names": b.get(
                "target_names",
                ["beta_motion", "beta_stability", "beta_energy"],
            ),
            "checkpoint_format": "objective_selector_v2",
        }

    # Objective Selector v1 compatibility.
    input_dim = int(b.get("input_dim", 15))
    output_dim = int(b.get("output_dim", 3))
    hidden = int(b.get("hidden", 64))

    m = _ObjectiveSelectorMLPv2(
        input_dim=input_dim,
        hidden=hidden,
        output_dim=output_dim,
    ).to(device)

    state = b.get("model_state_dict", b.get("state_dict", None))
    if state is None:
        raise KeyError(
            "objective checkpoint has no model_state_dict/state_dict keys; "
            f"available keys={list(b.keys())}"
        )

    m.load_state_dict(state)
    m.eval()

    return {
        "model": m,
        "bundle": b,
        "beta_names": b.get("beta_names", ["motion", "stability", "energy"]),
        "x_mean": torch.tensor(
            b.get("x_mean", [0.0] * input_dim),
            dtype=torch.float32,
            device=device,
        ),
        "x_std": torch.tensor(
            b.get("x_std", [1.0] * input_dim),
            dtype=torch.float32,
            device=device,
        ).clamp_min(1e-6),
        "feature_names": b.get("feature_names", []),
        "target_names": b.get(
            "target_names",
            ["beta_motion", "beta_stability", "beta_energy"],
        ),
        "checkpoint_format": "objective_selector_v1_compat",
    }


def load_ram(path: str, device):
    b = torch.load(path, map_location="cpu")
    m = RamInterventionMLP(
        int(b["input_dim"]),
        int(b["output_dim"]),
        int(b.get("hidden_dim", 256)),
    )
    m.load_state_dict(b["model_state_dict"])
    m.to(device).eval()
    return {
        "model": m,
        "bundle": b,
        "x_mean": torch.tensor(b["x_mean"], dtype=torch.float32),
        "x_std": torch.tensor(b["x_std"], dtype=torch.float32),
        "label_names": b["label_names"],
    }


def load_gms(path: str, device):
    b = torch.load(path, map_location="cpu")
    labels = b["label_names"]
    m = GMSMLP(
        int(b["input_dim"]),
        len(labels),
        int(b.get("hidden_dim", 128)),
    )
    m.load_state_dict(b["model_state_dict"])
    m.to(device).eval()
    return {
        "model": m,
        "bundle": b,
        "x_mean": torch.tensor(b["x_mean"], dtype=torch.float32),
        "x_std": torch.tensor(b["x_std"], dtype=torch.float32),
        "label_names": labels,
    }


class SupervisedStackServer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

        self.objective = load_objective(args.objective_model, self.device)
        self.ram = load_ram(args.ram_model, self.device)
        self.gms = load_gms(args.gms_model, self.device)

        self.n = 0
        self.t0 = time.time()

        print("[TRACER] supervised stack UDP server v2")
        print(f"[TRACER] device: {self.device}")
        print(f"[TRACER] bind:   {args.host}:{args.port}")
        print(f"[TRACER] objective: {args.objective_model}")
        print(f"[TRACER] ram:       {args.ram_model}")
        print(f"[TRACER] gms:       {args.gms_model}")
        print("[TRACER] ready", flush=True)

    def predict_objective(self, x: Optional[list]) -> Optional[Dict[str, Any]]:
        if x is None:
            return None

        expected = int(self.objective["bundle"]["input_dim"])
        if len(x) != expected:
            return {
                "ok": False,
                "error": f"objective_x length mismatch: got {len(x)}, expected {expected}",
            }

        with torch.no_grad():
            xn = tensor_norm(x, self.objective["x_mean"], self.objective["x_std"], self.device)
            beta = self.objective["model"](xn).cpu()[0].tolist()

        names = self.objective["beta_names"]
        return {
            "ok": True,
            "beta": beta,
            "beta_dict": {names[i]: beta[i] for i in range(len(names))},
        }

    def predict_ram(self, x: Optional[list]) -> Optional[Dict[str, Any]]:
        if x is None:
            return None

        expected = int(self.ram["bundle"]["input_dim"])
        if len(x) != expected:
            return {
                "ok": False,
                "error": f"ram_x length mismatch: got {len(x)}, expected {expected}",
            }

        with torch.no_grad():
            xn = tensor_norm(x, self.ram["x_mean"], self.ram["x_std"], self.device)
            probs = torch.sigmoid(self.ram["model"](xn)).cpu()[0].tolist()

        labels = self.ram["label_names"]
        pred = {labels[i]: probs[i] for i in range(len(labels))}

        # Compact fields useful for runtime logs.
        intervention_score = max(
            pred.get("future_gate_caution", 0.0),
            pred.get("future_gate_unstable", 0.0),
            pred.get("future_conservative_probe", 0.0),
            pred.get("future_low_speed", 0.0),
        )

        return {
            "ok": True,
            "labels": pred,
            "intervention_score": intervention_score,
            "future_override_mean": pred.get("future_override_mean", 0.0),
            "episode_success": pred.get("episode_success", 0.0),
        }

    def predict_gms(self, x: Optional[list]) -> Optional[Dict[str, Any]]:
        if x is None:
            return None

        expected = int(self.gms["bundle"]["input_dim"])
        if len(x) != expected:
            return {
                "ok": False,
                "error": f"gms_x length mismatch: got {len(x)}, expected {expected}",
            }

        with torch.no_grad():
            xn = tensor_norm(x, self.gms["x_mean"], self.gms["x_std"], self.device)
            logits = self.gms["model"](xn)
            probs = torch.softmax(logits, dim=-1).cpu()[0].tolist()

        labels = self.gms["label_names"]
        pred_id = int(max(range(len(probs)), key=lambda i: probs[i]))

        return {
            "ok": True,
            "label": labels[pred_id],
            "label_id": pred_id,
            "prob": probs[pred_id],
            "probs": {labels[i]: probs[i] for i in range(len(labels))},
        }

    def handle(self, req: Dict[str, Any]) -> Dict[str, Any]:
        self.n += 1

        resp = {
            "ok": True,
            "server": "tracer_supervised_stack_udp_server_v2",
            "n": self.n,
            "uptime_sec": time.time() - self.t0,
            "request_id": req.get("request_id", None),
        }

        try:
            resp["objective"] = self.predict_objective(req.get("objective_x", None))
            resp["ram"] = self.predict_ram(req.get("ram_x", None))
            resp["gms"] = self.predict_gms(req.get("gms_x", None))
        except Exception as e:
            resp["ok"] = False
            resp["error"] = repr(e)

        return resp

    def serve(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.args.host, self.args.port))

        while True:
            data, addr = sock.recvfrom(self.args.max_bytes)
            try:
                req = json.loads(data.decode("utf-8"))
            except Exception as e:
                resp = {"ok": False, "error": f"bad json: {e!r}"}
                sock.sendto(json.dumps(resp).encode("utf-8"), addr)
                continue

            resp = self.handle(req)
            sock.sendto(json.dumps(resp).encode("utf-8"), addr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50420)
    ap.add_argument("--max-bytes", type=int, default=65535)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--objective-model", default="artifacts/objective_selector_v2/objective_selector_v2.pt")
    ap.add_argument("--ram-model", default="artifacts/ram_intervention_v1/ram_intervention_v1.pt")
    ap.add_argument("--gms-model", default="artifacts/gms_classifier_v1/gms_classifier_v1.pt")

    args = ap.parse_args()

    srv = SupervisedStackServer(args)
    srv.serve()


if __name__ == "__main__":
    main()
