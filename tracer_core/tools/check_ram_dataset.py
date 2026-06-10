#!/usr/bin/env python
from __future__ import print_function

import sys
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python tracer_core/tools/check_ram_dataset.py <dataset.npz>")
        sys.exit(1)

    path = sys.argv[1]
    data = np.load(path)

    X = data["X"]
    Y = data["Y"]
    feature_fields = data["feature_fields"]
    label_fields = data["label_fields"]

    print("Dataset:", path)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("feature_fields:", list(feature_fields))
    print("label_fields:", list(label_fields))
    print("")

    print("X mean:", X.mean(axis=(0, 1)))
    print("X std: ", X.std(axis=(0, 1)))
    print("")
    print("Y mean:", Y.mean(axis=0))
    print("Y std: ", Y.std(axis=0))
    print("Y min: ", Y.min(axis=0))
    print("Y max: ", Y.max(axis=0))

    print("")
    print("First sample X[0, -1, :]:")
    print(X[0, -1, :])
    print("First sample Y[0]:")
    print(Y[0])


if __name__ == "__main__":
    main()
