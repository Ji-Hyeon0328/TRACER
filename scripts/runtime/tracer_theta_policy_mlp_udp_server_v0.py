#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn


class ThetaPolicyMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def get_nested(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def denorm(y: float, lo: float, hi: float) -> float:
    y = max(-1.0, min(1.0, float(y)))
    return lo + 0.5 * (y + 1.0) * (hi - lo)


def gate_code(level: str) -> float:
    return {"stable": 0.0, "caution": 1.0, "unstable": 2.0}.get(str(level), 0.0)


def action_code(action: str) -> float:
    return {
        "keep": 0.0,
        "would_cautious": 1.0,
        "would_conservative_probe": 2.0,
        "would_active_hold": 3.0,
        "would_recovery": 4.0,
    }.get(str(action), 0.0)


def feature_from_request(req: dict[str, Any], input_dim: int) -> list[float]:
    # UdpMetaGaitPolicy may send either a compact summary or nested policy_input.
    terrain = str(req.get("terrain", req.get("terrain_key", "unknown")))
    gait_mode = str(req.get("gait_mode", req.get("mode", "unknown")))

    beta = req.get("beta", {}) if isinstance(req.get("beta", {}), dict) else {}
    beta_v = float(req.get("beta_motion", beta.get("motion", 0.4)))
    beta_s = float(req.get("beta_stability", beta.get("stability", 0.4)))
    beta_e = float(req.get("beta_energy", beta.get("energy", 0.2)))

    ram = req.get("ram", {}) if isinstance(req.get("ram", {}), dict) else {}
    ram_level = str(req.get("ram_level", ram.get("ram_level", "stable")))
    ram_action = str(req.get("ram_gate_action", req.get("gate_action", ram.get("ram_gate_action", "keep"))))

    future_risk = float(req.get("future_risk", req.get("risk", 0.0)))
    future_slip = float(req.get("future_slip", 0.0))
    future_invalid = float(req.get("future_invalid", 0.0))
    fallen = float(req.get("fallen_prob", ram.get("fallen_prob", 0.0)))
    recovery = float(req.get("recovery_prob", ram.get("recovery_prob", 0.0)))
    sigma = float(req.get("sigma", ram.get("sigma", 0.0)))
    rho_norm = float(req.get("rho_norm", req.get("rho_norm_feature", 0.0)))

    # Terrain may not be in UDP request. Infer from gait mode when needed.
    if terrain == "unknown":
        if gait_mode == "fast":
            terrain = "flat_normal"
        elif gait_mode == "cautious_probe":
            terrain = "rough_mid"
        elif gait_mode == "high_clearance_slow_probe":
            terrain = "slope_5deg"

    terrain_onehot = [
        1.0 if terrain == "flat_normal" else 0.0,
        1.0 if terrain == "rough_mid" else 0.0,
        1.0 if terrain == "slope_5deg" else 0.0,
    ]

    x = []
    x.extend(terrain_onehot)
    x.extend([beta_v, beta_s, beta_e])
    x.extend([
        future_risk,
        future_slip,
        future_invalid,
        fallen,
        recovery,
        sigma,
        min(rho_norm / 10.0, 1.0),
        gate_code(ram_level) / 2.0,
        action_code(ram_action) / 2.0,
        float(req.get("would_override", 0.0)),
        float(req.get("vx", req.get("command_vx", 0.28))),
        float(req.get("body_height", 0.295)),
        float(req.get("swing_clearance", req.get("clearance", 0.03))),
    ])

    if len(x) < input_dim:
        x.extend([0.0] * (input_dim - len(x)))
    elif len(x) > input_dim:
        x = x[:input_dim]

    return x


def theta_to_response(theta: list[float], req: dict[str, Any]) -> dict[str, Any]:
    ranges = [
        (0.75, 1.35),
        (-0.12, 0.16),
        (0.35, 1.25),
        (-0.04, 0.06),
        (-0.04, 0.06),
        (0.00, 0.09),
        (0.70, 1.60),
        (0.00, 1.50),
    ]

    vals = [denorm(v, lo, hi) for v, (lo, hi) in zip(theta, ranges)]
    (
        gait_period_scale,
        duty_delta,
        step_length_scale,
        stance_width_delta,
        body_height_delta,
        clearance_delta,
        impedance_scale,
        residual_scale,
    ) = vals

    # UdpMetaGaitPolicy request does not include terrain/base command directly.
    # It does include gait_mode, beta, and RAM. Infer safe base defaults from gait_mode.
    gait_mode = str(req.get("gait_mode", req.get("mode", "normal")))

    if gait_mode == "fast":
        base_vx, base_h, base_clr = 0.28, 0.295, 0.030
    elif gait_mode == "cautious_probe":
        base_vx, base_h, base_clr = 0.055, 0.305, 0.050
    elif gait_mode == "high_clearance_slow_probe":
        base_vx, base_h, base_clr = 0.045, 0.315, 0.060
    elif gait_mode == "conservative":
        base_vx, base_h, base_clr = 0.035, 0.350, 0.110
    elif gait_mode in {"recovery", "avoid", "no_valid"}:
        base_vx, base_h, base_clr = 0.0, 0.330, 0.080
    else:
        base_vx, base_h, base_clr = 0.05, 0.335, 0.080

    # Preserve explicit request command if a future client sends it.
    base_vx = float(req.get("vx", req.get("command_vx", base_vx)))
    base_h = float(req.get("body_height", base_h))
    base_clr = float(req.get("swing_clearance", req.get("clearance", base_clr)))

    # Important:
    # step_length_scale is a gait parameter, not a direct command-vx multiplier.
    # For current runtime, keep the mode-level vx stable and use theta for richer
    # fields. Otherwise flat can be unintentionally slowed by theta dimensions.
    vx = max(0.0, min(0.28, base_vx))
    body_height = max(0.24, min(0.36, base_h + body_height_delta))
    swing_clearance = max(0.02, min(0.12, base_clr + clearance_delta))

    gait_period = max(0.24, min(0.75, 0.30 * gait_period_scale))
    duty_factor = max(0.45, min(0.78, 0.56 + duty_delta))
    step_length = max(0.0, min(0.16, vx * gait_period * step_length_scale))
    stance_width = max(0.18, min(0.32, 0.23 + stance_width_delta))

    enable = 0.0 if gait_mode in {"recovery", "avoid", "no_valid"} else 1.0

    meta = {
        "vx": vx,
        "yaw_rate": float(req.get("yaw_rate", 0.0)),
        "body_height": body_height,
        "swing_clearance": swing_clearance,
        "enable": enable,
        "gait_period": gait_period,
        "duty_factor": duty_factor,
        "step_length": step_length,
        "stance_width": stance_width,
        "impedance_scale": impedance_scale,
        "residual_gain_scale": residual_scale,
        "risk_scale": 1.0,
        "source_mode": gait_mode,
        "source_reason": "theta_policy_mlp_udp_v0",
        "extras": {
            "policy_type": "theta_mlp_udp_v0",
            "theta_norm": theta,
            "theta_physical": vals,
        },
    }

    return {
        "ok": True,
        "meta_gait": meta,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/theta_policy_mlp_v0/theta_policy_mlp_v0.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50310)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    model = ThetaPolicyMLP(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))

    print(f"[TRACER] theta MLP UDP server started: {args.host}:{args.port}")
    print(f"[TRACER] model={args.model} input_dim={ckpt['input_dim']} output_dim={ckpt['output_dim']}")

    n = 0
    while True:
        data, addr = sock.recvfrom(65535)
        try:
            req = json.loads(data.decode("utf-8"))
            x = feature_from_request(req, int(ckpt["input_dim"]))
            xt = torch.tensor([x], dtype=torch.float32)
            with torch.no_grad():
                theta = model(xt)[0].cpu().tolist()
            resp = theta_to_response(theta, req)
        except Exception as e:
            resp = {"ok": False, "error": repr(e)}

        sock.sendto(json.dumps(resp).encode("utf-8"), addr)
        n += 1
        if n == 1 or n % 50 == 0:
            print(f"[TRACER] served n={n} last_addr={addr} ok={resp.get('ok')}")


if __name__ == "__main__":
    main()
