#!/usr/bin/env python3

import argparse
import socket
import struct
import time

import numpy as np
import torch
import torch.nn as nn


class HighLevelPolicyV1MLP(nn.Module):
    def __init__(self, input_dim=39, output_dim=5, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/home/shoko/Tracer/TRACER/checkpoints/tracer_high_level_policy_v1_best.pt",
    )
    parser.add_argument("--udp-ip", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=50211)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print(f"[TRACER-V1] device: {device}")
    print(f"[TRACER-V1] loading checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model = HighLevelPolicyV1MLP(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)

    input_dim = int(ckpt["input_dim"])
    output_dim = int(ckpt["output_dim"])

    assert input_dim == 39, f"Expected v1 input_dim=39, got {input_dim}"
    assert output_dim == 5, f"Expected output_dim=5, got {output_dim}"

    input_fmt = "<39d"
    output_fmt = "<5d"
    input_size = struct.calcsize(input_fmt)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.udp_ip, args.udp_port))

    print(f"[TRACER-V1] learned high-level policy UDP server listening on {args.udp_ip}:{args.udp_port}")
    print("[TRACER-V1] input:  39 doubles = relative_goal[4] + context[16] + latent[16] + beta[3]")
    print("[TRACER-V1] output: 5 doubles = [vx, yaw_rate, body_height, clearance, enable]")

    last_log = 0.0

    while True:
        data, addr = sock.recvfrom(4096)

        if len(data) < input_size:
            now = time.time()
            if now - last_log > 1.0:
                print(f"[TRACER-V1] short packet from {addr}: {len(data)} bytes")
                last_log = now
            continue

        values = struct.unpack(input_fmt, data[:input_size])
        x = np.asarray(values, dtype=np.float32).reshape(1, 39)

        xn = (x - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)

        with torch.no_grad():
            xb = torch.from_numpy(xn).to(device)
            yn = model(xb).cpu().numpy()

        y = yn * y_std.reshape(1, -1) + y_mean.reshape(1, -1)
        y = y.reshape(-1).astype(np.float64)

        vx = clamp(float(y[0]), -0.30, 0.30)
        yaw_rate = clamp(float(y[1]), -0.60, 0.60)
        body_height = clamp(float(y[2]), 0.25, 0.36)
        clearance = clamp(float(y[3]), 0.00, 0.12)
        enable_raw = clamp(float(y[4]), 0.0, 1.0)
        enable = 1.0 if enable_raw > 0.5 else 0.0

        out = [vx, yaw_rate, body_height, clearance, enable]
        packet = struct.pack(output_fmt, *out)
        sock.sendto(packet, addr)

        now = time.time()
        if now - last_log > 1.0:
            rel = x.reshape(-1)[:4]
            beta = x.reshape(-1)[36:39]
            print(
                "[TRACER-V1] policy "
                f"rel=({rel[0]:.2f},{rel[1]:.2f},{rel[2]:.2f},{rel[3]:.1f}) "
                f"beta=({beta[0]:.2f},{beta[1]:.2f},{beta[2]:.2f}) "
                f"-> vx={vx:.3f} yaw={yaw_rate:.3f} "
                f"h={body_height:.3f} clr={clearance:.3f} "
                f"enable_raw={enable_raw:.3f} enable={enable:.1f}"
            )
            last_log = now


if __name__ == "__main__":
    main()
