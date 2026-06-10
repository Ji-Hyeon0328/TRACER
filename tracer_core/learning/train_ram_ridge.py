#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import json
import time

import numpy as np


def mse(y_pred, y_true):
    d = y_pred - y_true
    return float(np.mean(d * d))


def mae(y_pred, y_true):
    return float(np.mean(np.abs(y_pred - y_true)))


def main():
    if len(sys.argv) < 2:
        dataset_path = "/root/TRACER/datasets/ram/ram_goal_tracer_v0.npz"
    else:
        dataset_path = sys.argv[1]

    out_dir = "/root/TRACER/checkpoints/ram"
    if len(sys.argv) >= 3:
        out_dir = sys.argv[2]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = np.load(dataset_path)

    X = data["X"].astype(np.float64)
    Y = data["Y"].astype(np.float64)

    feature_fields = [str(x) for x in data["feature_fields"]]
    label_fields = [str(x) for x in data["label_fields"]]

    N, H, F = X.shape
    D = H * F

    Xf = X.reshape((N, D))

    rng = np.random.RandomState(17)
    idx = np.arange(N)
    rng.shuffle(idx)

    n_train = int(0.8 * N)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_train = Xf[train_idx]
    Y_train = Y[train_idx]

    X_val = Xf[val_idx]
    Y_val = Y[val_idx]

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std < 1e-8] = 1.0

    y_mean = Y_train.mean(axis=0)
    y_std = Y_train.std(axis=0)
    y_std[y_std < 1e-8] = 1.0

    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std

    Y_train_n = (Y_train - y_mean) / y_std
    Y_val_n = (Y_val - y_mean) / y_std

    # Add bias column.
    ones_train = np.ones((X_train_n.shape[0], 1), dtype=np.float64)
    ones_val = np.ones((X_val_n.shape[0], 1), dtype=np.float64)

    A_train = np.concatenate([X_train_n, ones_train], axis=1)
    A_val = np.concatenate([X_val_n, ones_val], axis=1)

    # Ridge regression:
    # W = (A^T A + lambda I)^-1 A^T Y
    # Do not regularize bias too strongly.
    ridge_lambda = 1e-3
    I = np.eye(A_train.shape[1], dtype=np.float64)
    I[-1, -1] = 0.0

    lhs = np.dot(A_train.T, A_train) + ridge_lambda * I
    rhs = np.dot(A_train.T, Y_train_n)

    W = np.linalg.solve(lhs, rhs)

    pred_train_n = np.dot(A_train, W)
    pred_val_n = np.dot(A_val, W)

    pred_train = pred_train_n * y_std + y_mean
    pred_val = pred_val_n * y_std + y_mean

    # Clamp to proxy RAM output range.
    pred_train_clamped = np.clip(pred_train, 0.0, 1.0)
    pred_val_clamped = np.clip(pred_val, 0.0, 1.0)

    train_mse = mse(pred_train_clamped, Y_train)
    val_mse = mse(pred_val_clamped, Y_val)

    train_mae = mae(pred_train_clamped, Y_train)
    val_mae = mae(pred_val_clamped, Y_val)

    per_label_val_mae = np.mean(np.abs(pred_val_clamped - Y_val), axis=0)
    per_label_val_mse = np.mean((pred_val_clamped - Y_val) ** 2, axis=0)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(out_dir, "ram_ridge_v0_%s.npz" % stamp)
    stats_path = os.path.join(out_dir, "ram_ridge_v0_%s_stats.json" % stamp)

    np.savez(
        model_path,
        W=W,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        history_len=np.asarray([H]),
        feature_dim=np.asarray([F]),
        feature_fields=np.asarray(feature_fields),
        label_fields=np.asarray(label_fields),
        ridge_lambda=np.asarray([ridge_lambda])
    )

    stats = {
        "dataset_path": dataset_path,
        "num_samples": int(N),
        "history_len": int(H),
        "feature_dim": int(F),
        "flat_dim": int(D),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "feature_fields": feature_fields,
        "label_fields": label_fields,
        "ridge_lambda": ridge_lambda,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "per_label_val_mae": [float(x) for x in per_label_val_mae],
        "per_label_val_mse": [float(x) for x in per_label_val_mse],
        "model_path": model_path
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print("Wrote model:", model_path)
    print("Wrote stats:", stats_path)
    print("")
    print("N=%d H=%d F=%d D=%d" % (N, H, F, D))
    print("train samples:", len(train_idx))
    print("val samples:", len(val_idx))
    print("")
    print("train_mse:", train_mse)
    print("val_mse:", val_mse)
    print("train_mae:", train_mae)
    print("val_mae:", val_mae)
    print("")
    for i, label in enumerate(label_fields):
        print("%s val_mae=%.6f val_mse=%.6f" % (
            label,
            per_label_val_mae[i],
            per_label_val_mse[i]
        ))

    print("")
    print("Example predictions:")
    for k in range(min(5, len(val_idx))):
        yt = Y_val[k]
        yp = pred_val_clamped[k]
        print("  true=[%.4f %.4f] pred=[%.4f %.4f]" % (
            yt[0], yt[1], yp[0], yp[1]
        ))


if __name__ == "__main__":
    main()
