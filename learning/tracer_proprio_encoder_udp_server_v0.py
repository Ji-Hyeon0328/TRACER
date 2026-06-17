#!/usr/bin/env python3

import argparse
import socket
import struct
import time

import numpy as np
import torch
import torch.nn as nn


class ProprioEncoderMLP(nn.Module):
    def __init__(self, input_dim=45, output_dim=32, hidden_dim=128):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/home/shoko/Tracer/TRACER/checkpoints/tracer_proprio_encoder_v0_best.pt",
    )
    parser.add_argument("--udp-ip", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=50200)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[TRACER] device: {device}")
    print(f"[TRACER] loading checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model = ProprioEncoderMLP(
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

    input_fmt = "<45d"
    output_fmt = "<32d"
    input_size = struct.calcsize(input_fmt)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.udp_ip, args.udp_port))

    print(f"[TRACER] learned proprio encoder UDP server listening on {args.udp_ip}:{args.udp_port}")
    print("[TRACER] input:  45 doubles proprio_vector")
    print("[TRACER] output: 32 doubles [context(16), latent(16)]")

    last_log = 0.0

    while True:
        data, addr = sock.recvfrom(4096)

        if len(data) < input_size:
            now = time.time()
            if now - last_log > 1.0:
                print(f"[TRACER] short packet from {addr}: {len(data)} bytes")
                last_log = now
            continue

        values = struct.unpack(input_fmt, data[:input_size])
        x = np.asarray(values, dtype=np.float32).reshape(1, 45)

        xn = (x - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)

        with torch.no_grad():
            xb = torch.from_numpy(xn).to(device)
            yn = model(xb).cpu().numpy()

        y = yn * y_std.reshape(1, -1) + y_mean.reshape(1, -1)
        y = y.reshape(-1).astype(np.float64)

        packet = struct.pack(output_fmt, *[float(v) for v in y])
        sock.sendto(packet, addr)

        now = time.time()
        if now - last_log > 1.0:
            context = y[:16]
            latent = y[16:32]
            print(
                "[TRACER] inference "
                f"context h={context[0]:.3f} roll={context[1]:.3f} pitch={context[2]:.3f} "
                f"speed={context[6]:.3f} contact={context[8]:.3f} | "
                f"latent risk={latent[0]:.3f} low_contact={latent[1]:.3f} slip={latent[2]:.3f}"
            )
            last_log = now


if __name__ == "__main__":
    main()
