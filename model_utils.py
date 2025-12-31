"""Model save/load utilities for numpy-based prototype.

We store weights, biases, and minimal metadata in a compressed npz with
explicit keys: w0, w1, ..., b0, b1, ..., and a JSON meta string.
"""
import json
import numpy as np


def save_model_weights(path: str, weights: list, biases: list, meta: dict):
    payload = {}
    for i, w in enumerate(weights):
        payload[f"w{i}"] = w
    for i, b in enumerate(biases):
        payload[f"b{i}"] = b
    payload["meta_json"] = json.dumps(meta)
    np.savez_compressed(path, **payload)


def load_model_weights(path: str):
    data = np.load(path, allow_pickle=True)
    weights = []
    biases = []
    meta = {}
    for k in sorted(data.files):
        if k.startswith("w"):
            weights.append(data[k])
        elif k.startswith("b"):
            biases.append(data[k])
        elif k == "meta_json":
            try:
                meta = json.loads(str(data[k]))
            except Exception:
                meta = {}
    return weights, biases, meta
