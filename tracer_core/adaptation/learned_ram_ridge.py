#!/usr/bin/env python
from __future__ import print_function

import os
import numpy as np


MODE_TO_ID = {
    "nominal": 0,
    "cautious_mismatch": 1,
    "conservative_mismatch": 2,
    "conservative_posture": 3,
    "recovery": 4,
    "reached": 5,
    "": -1,
}


class LearnedRamRidge(object):
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise RuntimeError("LearnedRamRidge model not found: %s" % model_path)

        data = np.load(model_path)

        self.W = data["W"]
        self.x_mean = data["x_mean"]
        self.x_std = data["x_std"]
        self.y_mean = data["y_mean"]
        self.y_std = data["y_std"]

        self.history_len = int(data["history_len"][0])
        self.feature_dim = int(data["feature_dim"][0])

        self.feature_fields = [str(x) for x in data["feature_fields"]]
        self.label_fields = [str(x) for x in data["label_fields"]]

        self.buffer = []

    def reset(self):
        self.buffer = []

    def get_feature(self, obs, field):
        if field == "vx_mod_error":
            return float(obs.get("raw_vx_axis", 0.0)) - float(obs.get("vx_axis", 0.0))

        if field == "yaw_mod_error":
            return float(obs.get("raw_yaw_axis", 0.0)) - float(obs.get("yaw_axis", 0.0))

        if field == "mode_id":
            return float(MODE_TO_ID.get(obs.get("mode", ""), -1))

        return float(obs.get(field, 0.0))

    def update(self, obs):
        feat = []
        for field in self.feature_fields:
            feat.append(self.get_feature(obs, field))

        self.buffer.append(feat)

        if len(self.buffer) > self.history_len:
            self.buffer = self.buffer[-self.history_len:]

    def ready(self):
        return len(self.buffer) >= self.history_len

    def predict(self):
        if not self.ready():
            return None

        X = np.asarray(self.buffer[-self.history_len:], dtype=np.float64)

        if X.shape[0] != self.history_len or X.shape[1] != self.feature_dim:
            return None

        Xf = X.reshape((1, self.history_len * self.feature_dim))
        Xn = (Xf - self.x_mean) / self.x_std

        A = np.concatenate([Xn, np.ones((1, 1), dtype=np.float64)], axis=1)

        pred_n = np.dot(A, self.W)
        pred = pred_n * self.y_std + self.y_mean
        pred = np.clip(pred, 0.0, 1.0)

        out = {}
        for i, label in enumerate(self.label_fields):
            out[label] = float(pred[0, i])

        return out
