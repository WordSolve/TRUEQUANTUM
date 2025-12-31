# TRUEQUANTUM
Prototype: Zephyr Ultimate AI Miner (Monero-compatible, safe demo)

This repository provides an educational prototype of the "Zephyr Ultimate AI Miner" concept with **Monero (XMR) mining support**. It includes a light quantum-inspired nonce sampler, a numpy-based prototype neural network, simple feature extraction, QuantumGuardQC3 advanced hashing components, and a telemetry-only XMRig API utility. The code is explicitly for research and testing — it does not automatically perform real mining or submit shares unless you configure and run a real miner (XMRig) yourself.

## Monero Support
- **Mining algorithm**: CryptoNight, CryptoNightR, RandomX (handled by XMRig)
- **Wallet format**: Monero mainnet addresses start with '4', stagenet with '8', testnet with '9'
- **Default pools**: supportxmr.com, minexmr.com, moneropools.com
- **XMRig API port**: 18081 (default for Monero)

Files
- `zephyr_ultimate_ai_miner_fixed.py` — Monero-compatible prototype miner with quantum sampler, neural network, and QC3 advanced hashing components.
- `quantum_components.py` — QuantumGuardQC3 advanced hashing: Superposition, 10D Layer Cake, MAC 10, 9-Code, Tornado Convergence, Singularity, 9D Updraft Tornado.
- `demo_no_deps.py` — Pure-Python demo that runs in restricted containers (no external deps).
- `xmrig_telemetry.py` — Non-invasive telemetry tool to query XMRig's HTTP API.
- `xmrig_control.py` — Best-effort helpers to start/stop XMRig locally and attempt API share submission.
- `dashboard.py` — Flask dashboard with Monero wallet display, XMRig control, telemetry polling, and cipher chatter.
- `requirements.txt` — Minimal Python dependencies: `numpy`, `requests`, `Flask`.

Safety & legal note
- This code is educational. If you enable real mining or pool submission, make sure you:
	- Have the legal right to mine in your jurisdiction.
	- Understand electricity costs and environmental impact.
	- Follow pool and wallet provider terms of service.
	- Use a valid Monero wallet address (mainnet: starts with '4').

Getting started (recommended workflow)

1) Prepare an isolated Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) Quick demo (no external dependencies)

```bash
python demo_no_deps.py --cycles 5
```

3) Full prototype (requires `numpy` and `requests`)

Edit `zephyr_ultimate_ai_miner_fixed.py`'s `CONFIG` at the top to set your Monero wallet, pool, and XMRig API URL. Example:

```python
CONFIG = {
    "POOL_URL": os.environ.get("MONERO_POOL_URL", "pool.supportxmr.com"),
    "POOL_PORT": int(os.environ.get("MONERO_POOL_PORT", "3333")),
    "WALLET_ADDRESS": os.environ.get("MONERO_WALLET", "4BrL51JCzqkYjMCJ5ch2XUUoJGMVMyJUUbYodQyonmSEZAZvDZviiD3fGV61jCJoNroxPJS2XH8kvMQeFqBED76m4539A6o"),
    ...
}
```

Or set environment variables:

```bash
export MONERO_WALLET="<your-monero-address>"
export MONERO_POOL="pool.supportxmr.com"
export MONERO_POOL_PORT="3333"
export XMRIG_API_URL="http://127.0.0.1:18081"
```

Then run (simulation mode recommended first):

```bash
python zephyr_ultimate_ai_miner_fixed.py --simulate --cycles 20
```

4) Dashboard (web UI)

```bash
export MONERO_WALLET="<your-address>"
python dashboard.py --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 to manage XMRig process and view telemetry.

5) Telemetry-only connectivity to XMRig's HTTP API (non-invasive GET):

```bash
python xmrig_telemetry.py --api-url http://127.0.0.1:18081

# or via main script
python zephyr_ultimate_ai_miner_fixed.py --telemetry --telemetry-url http://127.0.0.1:8080
```

XMRig control (best-effort; you must provide a valid XMRig binary and config)

```bash
# Start XMRig as a subprocess using CONFIG values
python zephyr_ultimate_ai_miner_fixed.py --start-xmrig --simulate --cycles 5

# Stop the XMRig subprocess started above
python zephyr_ultimate_ai_miner_fixed.py --stop-xmrig --simulate --cycles 1

# Attempt share submission via XMRig HTTP API (only if your XMRig build exposes /submit)
python zephyr_ultimate_ai_miner_fixed.py --cycles 5  # requires ENABLE_XMRIG_API True and XMRIG_API_URL set
```
Notes:
- Control and submission endpoints vary by XMRig version; failures will be printed but won’t crash the script.
- This project will not auto-start XMRig unless you pass `--start-xmrig` with a valid binary path in CONFIG.

4) Dashboard (simulated hashing UI, black/green)

```bash
python dashboard.py --host 0.0.0.0 --port 8000
# open http://localhost:8000/
```
Notes: the dashboard only hashes random nonces locally and shows a hardcoded Cake wallet. It does not submit shares or control XMRig. You can poll XMRig telemetry from the UI; it remains non-invasive.

XMRig setup (brief)
- Download XMRig from: https://github.com/xmrig/xmrig/releases
- Run XMRig locally or on a host and enable its HTTP API (see XMRig docs): typically with `--api 127.0.0.1:8080` or equivalent config.
- Start XMRig separately; this repo will not start or manage the XMRig process for you.

CONFIG example (paste into `zephyr_ultimate_ai_miner_fixed.py` and edit):

```python
CONFIG = {
		"POOL_URL": "your-pool-url.com",
		"POOL_PORT": 1123,
		"WALLET_ADDRESS": "your-wallet-address",
		"WORKER_NAME": "your-worker-name",
		"XMRIG_API_URL": "http://127.0.0.1:8080",
		"ENABLE_XMRIG_API": True,
		"ENABLE_QUANTUM_AI": True,
		"ENABLE_NEURAL_NETWORKS": True,
		"QUANTUM_QUBITS": 32,
		"NEURAL_LAYERS": [128,256,512,256,128,64,32,16,8,1],
		"LEARNING_RATE": 0.001,
		"NUM_THREADS": os.cpu_count(),
}
```

Telemetry and monitoring
- Use `xmrig_telemetry.py` to query XMRig's HTTP API without affecting miner state. It will print JSON status if reachable.
- The full prototype (`zephyr_ultimate_ai_miner_fixed.py`) collects internal AI stats and prints simple reports during execution; for production-style telemetry integrate a logging or metrics backend (Prometheus, Influx, etc.).

Training & convergence guidance (practical notes)

This prototype implements a very small online training step; if you want real convergence and meaningful neural models, follow these steps:

- Data collection
	- Collect labeled samples: features from candidate nonces paired with a binary label indicating success (share accepted / block found). Real labels require running against a real miner or a dataset of successful nonces.
	- Store samples centrally (CSV, Parquet, or a lightweight DB) and include timestamp, difficulty, pool response, thread id, and system stats.

- Model architecture and hyperparameters
	- Start with the prototype architecture: input 128 → hidden layers [128,256,512,256,128,64,32,16,8] → 1 output sigmoid.
	- Learning rate: 1e-3 as starting point. Use adaptive optimizers (Adam) or SGD with momentum (0.9) for stability.
	- Batch size: 32–512 depending on memory. For online learning, accumulate small batches (e.g., 32) and update frequently.
	- Regularization: use dropout (0.2–0.5) and L2 weight decay (1e-5–1e-3) to avoid overfitting.

- Training procedure
	- Split data into train/validation (e.g., 80/20). Monitor validation loss and AUC for convergence.
	- Early stopping: stop when validation loss doesn't improve for N epochs (N=5–10).
	- Learning rate schedule: reduce LR on plateau (factor 0.1) or use cosine decay.
	- Checkpoints: save model weights every epoch or every N updates and keep the best validation model.

- Convergence diagnostics
	- Plot training and validation loss curves, ROC AUC, and precision/recall over time.
	- Verify model calibration — predicted probabilities should match observed success rates (calibration curve).
	- If the model overfits quickly, increase regularization or collect more diverse data.

- Evaluation metrics
	- Primary: ROC AUC, precision@K (top-K candidate success rate), calibration error.
	- Secondary: impact on actual mining throughput (after careful A/B testing on a single thread or in simulation).

Hardware and performance tips
- CPU-based training: use numpy + multithreading or small PyTorch/TensorFlow CPU builds.
- For production speed, use a lightweight ML runtime (ONNX, quantized models) for per-candidate scoring.
- Keep feature extraction fast and memory-efficient; avoid recomputing expensive features per nonce.

Integration checklist (before enabling real mining)
1. Verify XMRig runs and exposes HTTP API (`xmrig_telemetry.py` should reach it).
2. Verify that `zephyr_ultimate_ai_miner_fixed.py --simulate` runs without errors and prints expected logs.
3. Configure `CONFIG` carefully with your pool and wallet; keep `ENABLE_XMRIG_API` True only when API is reachable and you intend to use telemetry.
4. Test model predictions in simulation mode for stability before allowing any automatic submission or control of miner parameters.

Troubleshooting
- `ModuleNotFoundError: No module named 'numpy'` — create and activate a venv and `pip install -r requirements.txt`.
- `Connection refused` from `xmrig_telemetry.py` — ensure XMRig is running and its API is bound to the same host/port.

Model training and loading workflow (prototype)

```bash
# 1) generate synthetic samples (replace with real labels for meaningful models)
python data_collector.py --out samples.npz --n 2000

# 2) train and save weights with metadata
python train_model.py --data samples.npz --out model_weights.npz --hidden 128,64 --epochs 10 --lr 0.001

# 3) run miner prototype loading the saved weights
python zephyr_ultimate_ai_miner_fixed.py --simulate --cycles 50 --use-model model_weights.npz
```

Notes
- The loader in `zephyr_ultimate_ai_miner_fixed.py` will rebuild the neural net if the saved metadata contains the layer sizes and then apply the weights (shape-checked).
- For real performance, train with real labeled data collected from actual mining outcomes; the provided data_collector/train scripts are for demonstration.

Extending the project
- Add dataset collection and a small training script that uses PyTorch/TensorFlow for robust training.
- Add a telemetry/metrics backend for long-term monitoring.
- Replace the prototype numpy NN with a saved model (ONNX) for faster inference.

Contact / Contribute
- This is a prototype. For feature requests or contributions, open an issue or submit a PR.
