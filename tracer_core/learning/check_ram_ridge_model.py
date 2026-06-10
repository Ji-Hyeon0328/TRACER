#!/usr/bin/env python
from __future__ import print_function

import sys
import numpy as np


def main():
    if len(sys.argv) < 3:
        print("Usage: python tracer_core/learning/check_ram_ridge_model.py <model.npz> <dataset.npz>")
        sys.exit(1)

    model_path = sys.argv[1]
    dataset_path = sys.argv[2]

    model = np.load(model_path)
    data = np.load(dataset_path)

    W = model["W"]
    x_mean = model["x_mean"]
    x_std = model["x_std"]
    y_mean = model["y_mean"]
    y_std = model["y_std"]

    X = data["X"].astype(np.float64)
    Y = data["Y"].astype(np.float64)

    N = X.shape[0]
    Xf = X.reshape((N, -1))

    Xn = (Xf - x_mean) / x_std
    A = np.concatenate([Xn, np.ones((N, 1))], axis=1)

    pred_n = np.dot(A, W)
    pred = pred_n * y_std + y_mean
    pred = np.clip(pred, 0.0, 1.0)

    err = np.abs(pred - Y)

    print("Model:", model_path)
    print("Dataset:", dataset_path)
    print("N:", N)
    print("")
    print("MAE:", err.mean(axis=0))
    print("MSE:", ((pred - Y) ** 2).mean(axis=0))
    print("")
    print("Prediction range:")
    print("  pred min:", pred.min(axis=0))
    print("  pred max:", pred.max(axis=0))
    print("  true min:", Y.min(axis=0))
    print("  true max:", Y.max(axis=0))
    print("")
    print("First 10 predictions:")
    for i in range(min(10, N)):
        print("%04d true=[%.4f %.4f] pred=[%.4f %.4f] err=[%.4f %.4f]" % (
            i,
            Y[i, 0], Y[i, 1],
            pred[i, 0], pred[i, 1],
            err[i, 0], err[i, 1]
        ))


if __name__ == "__main__":
    main()
