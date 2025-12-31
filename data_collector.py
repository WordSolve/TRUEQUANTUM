"""Data collection helper: generate or collect labeled samples for training.

This script can run in simulation mode to produce a `samples.npz` of features
and labels that `train_model.py` can use. Real collection should record
features against real miner outcomes.
"""
import argparse
import time
import random
import numpy as np


def generate_samples(n_samples=1000):
    X = []
    Y = []
    for _ in range(n_samples):
        nonce = random.getrandbits(32)
        # simple synthetic features: bitcounts and low bytes
        bits = bin(nonce)[2:].zfill(32)[-32:]
        bit_array = [int(b) for b in bits]
        low_bytes = [((nonce >> (8 * i)) & 0xFF) / 255.0 for i in range(4)]
        feat = bit_array + low_bytes
        label = 1.0 if (nonce % 97 == 0) else 0.0
        X.append(feat[:128] + [0.0] * max(0, 128 - len(feat)))
        Y.append(label)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples.npz")
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()
    X, Y = generate_samples(args.n)
    np.savez(args.out, X=X, Y=Y)
    print(f"Saved {len(Y)} samples to {args.out}")


if __name__ == "__main__":
    main()
