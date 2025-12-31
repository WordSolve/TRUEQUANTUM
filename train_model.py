"""Train a simple numpy-based fully-connected model on collected samples.

This is a minimal example for demonstration and checkpointing; use a proper
framework (PyTorch/TensorFlow) for real training.
"""
import argparse
import numpy as np

from model_utils import save_model_weights


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train(X, Y, layers, lr=1e-3, epochs=10, batch_size=32):
    # layers: full architecture list, e.g., [input_dim, 128, 64, 1]
    params_w = []
    params_b = []
    for i in range(len(layers) - 1):
        in_dim, out_dim = layers[i], layers[i + 1]
        W = np.random.randn(in_dim, out_dim) * 0.01
        b = np.zeros(out_dim)
        params_w.append(W)
        params_b.append(b)

    n = X.shape[0]
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        X_shuf = X[perm]
        Y_shuf = Y[perm]
        for i in range(0, n, batch_size):
            xb = X_shuf[i : i + batch_size]
            yb = Y_shuf[i : i + batch_size]
            activations = [xb]
            # forward
            for li in range(len(params_w) - 1):
                xb = np.tanh(xb.dot(params_w[li]) + params_b[li])
                activations.append(xb)
            logits = xb.dot(params_w[-1]) + params_b[-1]
            preds = sigmoid(logits).ravel()
            error = preds - yb
            # backprop last layer
            grad_W_last = activations[-1].T.dot(error.reshape(-1, 1)) / xb.shape[0]
            grad_b_last = error.mean()
            params_w[-1] -= lr * grad_W_last
            params_b[-1] -= lr * grad_b_last
            # backprop hidden layers
            grad = error.reshape(-1, 1).dot(params_w[-1].T)
            for li in reversed(range(len(params_w) - 1)):
                act = activations[li + 1]
                grad_act = (1 - act * act) * grad
                grad_W = activations[li].T.dot(grad_act) / activations[li].shape[0]
                grad_b = grad_act.mean(axis=0)
                params_w[li] -= lr * grad_W
                params_b[li] -= lr * grad_b
                if li > 0:
                    grad = grad_act.dot(params_w[li].T)
        # epoch eval
        h = X
        for li in range(len(params_w) - 1):
            h = np.tanh(h.dot(params_w[li]) + params_b[li])
        p_all = sigmoid(h.dot(params_w[-1]) + params_b[-1]).ravel()
        loss = -np.mean(Y * np.log(np.clip(p_all, 1e-8, 1.0)) + (1 - Y) * np.log(np.clip(1 - p_all, 1e-8, 1.0)))
        print(f"Epoch {epoch+1}/{epochs} loss={loss:.4f}")

    return params_w, params_b


def parse_hidden(arg: str):
    if not arg:
        return [128, 64]
    return [int(x) for x in arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="samples.npz")
    parser.add_argument("--out", default="model_weights.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", default="128,64", help="Comma list of hidden sizes")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    data = np.load(args.data)
    X = data["X"]
    Y = data["Y"]
    hidden = parse_hidden(args.hidden)
    layers = [X.shape[1]] + hidden + [1]
    weights, biases = train(X, Y, layers=layers, lr=args.lr, epochs=args.epochs)
    meta = {"layers": layers, "lr": args.lr}
    save_model_weights(args.out, weights, biases, meta)
    print(f"Saved model weights to {args.out} with meta {meta}")


if __name__ == "__main__":
    main()
