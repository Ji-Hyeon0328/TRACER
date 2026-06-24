#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts/tracer_ram_v0"

HOST = os.environ.get("TRACER_RAM_UDP_HOST", "127.0.0.1")
PORT = int(os.environ.get("TRACER_RAM_UDP_PORT", "50210"))


class RAMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, latent_dim: int = 16, num_heads: int = 6):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.rho_head = nn.Linear(hidden_dim, latent_dim)
        self.sigma_head = nn.Linear(hidden_dim, latent_dim)
        self.pred_trunk = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.cls_head = nn.Linear(64, num_heads)
        self.score_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        _, h = self.encoder(x)
        h = h[-1]

        rho = self.rho_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-4

        z = self.pred_trunk(rho)
        logits = self.cls_head(z)
        score = self.score_head(z).squeeze(-1)

        return rho, sigma, logits, score


def load_ram():
    meta_path = ARTIFACT_DIR / "scaler_metrics_env.json"
    model_path = ARTIFACT_DIR / "model.pt"

    meta = json.loads(meta_path.read_text())
    ckpt = torch.load(model_path, map_location="cpu")

    input_dim = int(ckpt["input_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    latent_dim = int(ckpt["latent_dim"])
    head_names = list(ckpt["head_names"])

    model = RAMNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_heads=len(head_names),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_mean = np.asarray(meta["x_mean"], dtype=np.float32)
    x_std = np.asarray(meta["x_std"], dtype=np.float32)
    x_std[x_std < 1e-6] = 1.0

    score_mean = float(meta["score_mean"])
    score_std = float(meta["score_std"])
    window = int(meta["window"])

    return model, head_names, x_mean, x_std, score_mean, score_std, window, input_dim


def infer(model, head_names, x_mean, x_std, score_mean, score_std, window, input_dim, payload):
    x = np.asarray(payload["window"], dtype=np.float32)

    if x.shape != (window, input_dim):
        raise ValueError(f"expected window shape {(window, input_dim)}, got {x.shape}")

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - x_mean[None, :]) / x_std[None, :]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    abs_z = np.abs(x)
    debug = {
        "z_abs_mean": float(abs_z.mean()),
        "z_abs_max": float(abs_z.max()),
        "z_abs_p95": float(np.percentile(abs_z, 95)),
        "last_z_abs_mean": float(np.abs(x[-1]).mean()),
        "last_z_abs_max": float(np.abs(x[-1]).max()),
    }

    xt = torch.from_numpy(x[None, :, :]).float()

    with torch.no_grad():
        rho, sigma, logits, score_norm = model(xt)
        probs = torch.sigmoid(logits)

    rho_np = rho[0].cpu().numpy().astype(float)
    sigma_np = sigma[0].cpu().numpy().astype(float)
    probs_np = probs[0].cpu().numpy().astype(float)

    score = float(score_norm.item() * score_std + score_mean)
    sigma_mean = float(np.mean(sigma_np))
    rho_norm = float(np.linalg.norm(rho_np))

    head_probs = {name: float(prob) for name, prob in zip(head_names, probs_np)}

    return {
        "ok": True,
        "head_names": head_names,
        "head_probs": head_probs,
        "rho": rho_np.tolist(),
        "sigma": sigma_np.tolist(),
        "sigma_mean": sigma_mean,
        "rho_norm": rho_norm,
        "style_score_pred": score,
        "input_debug": debug,
        "server_time": time.time(),
    }


def main():
    model, head_names, x_mean, x_std, score_mean, score_std, window, input_dim = load_ram()

    print("[TRACER RAM SERVER] artifact:", ARTIFACT_DIR)
    print("[TRACER RAM SERVER] bind:", HOST, PORT)
    print("[TRACER RAM SERVER] window:", window, "input_dim:", input_dim)
    print("[TRACER RAM SERVER] heads:", head_names, flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))

    while True:
        data, addr = sock.recvfrom(65535)
        try:
            payload = json.loads(data.decode("utf-8"))
            result = infer(
                model,
                head_names,
                x_mean,
                x_std,
                score_mean,
                score_std,
                window,
                input_dim,
                payload,
            )
        except Exception as e:
            result = {
                "ok": False,
                "error": repr(e),
                "server_time": time.time(),
            }

        sock.sendto(json.dumps(result).encode("utf-8"), addr)


if __name__ == "__main__":
    main()
