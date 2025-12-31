"""QuantumGuardQC3 Dashboard (Monero-compatible, real telemetry, optional XMRig control)

Features:
- Monero wallet address display (configurable via environment).
- Telemetry polling of XMRig HTTP API (non-invasive GET).
- Optional start/stop of a local XMRig process (best-effort) using env/defaults for Monero mining.
- AI cipher chatter panel (cosmetic).

Run:
  python dashboard.py --host 0.0.0.0 --port 8000

Then open http://localhost:8000/
"""
import argparse
import os
import threading
import time

from flask import Flask, redirect, render_template_string, request, url_for

try:
    import requests
except Exception:
    requests = None

try:
    from xmrig_control import start_xmrig, stop_process
except Exception:
    start_xmrig = None
    stop_process = None

# Monero wallet (mainnet address starts with '4')
# Example address shown; configure with environment variable MONERO_WALLET
WALLET_ADDRESS = os.environ.get(
    "MONERO_WALLET",
    "4BrL51JCzqkYjMCJ5ch2XUUoJGMVMyJUUbYodQyonmSEZAZvDZviiD3fGV61jCJoNroxPJS2XH8kvMQeFqBED76m4539A6o"
)

# Defaults (override with environment)
DEFAULT_API_URL = os.environ.get("XMRIG_API_URL", "http://127.0.0.1:18081")  # Default XMRig API port
DEFAULT_POOL = os.environ.get("MONERO_POOL", "pool.supportxmr.com")  # Popular Monero pool
DEFAULT_PORT = int(os.environ.get("MONERO_POOL_PORT", "3333"))  # Standard Monero pool port
DEFAULT_WORKER = os.environ.get("MONERO_WORKER", "quantumguard-qc3")  # Worker name
DEFAULT_XMRIG_PATH = os.environ.get("XMRIG_PATH", "xmrig")  # Path to xmrig binary

app = Flask(__name__)
XM_PROC = None
XM_LOCK = threading.Lock()


def fetch_telemetry(api_url: str, timeout: float = 1.5):
    if not requests:
        return False, "requests not available"
    try:
        r = requests.get(api_url, timeout=timeout)
        r.raise_for_status()
        try:
            return True, r.json()
        except Exception:
            return True, r.text
    except Exception as e:
        return False, str(e)


def parse_telemetry(payload):
    """Extract hashrate and share info from XMRig telemetry JSON."""
    if not isinstance(payload, dict):
        return None, None, None
    
    hashrate = None
    shares_accepted = 0
    shares_rejected = 0
    
    # Parse hashrate from XMRig API
    if "hashrate" in payload:
        hashrate_data = payload["hashrate"]
        if isinstance(hashrate_data, dict) and "total" in hashrate_data:
            hashrate_list = hashrate_data["total"]
            if isinstance(hashrate_list, list) and len(hashrate_list) > 0:
                hashrate = hashrate_list[0]  # Current hashrate (H/s)
    
    # Parse accepted/rejected shares
    if "results" in payload:
        results = payload["results"]
        if isinstance(results, dict):
            shares_accepted = results.get("shares_good", 0)
            shares_rejected = results.get("shares_total", 0) - shares_accepted if "shares_total" in results else 0
    
    return hashrate, shares_accepted, shares_rejected


def estimate_daily_earnings(hashrate_hs):
    """Rough estimate of daily XMR earnings at current difficulty."""
    if not hashrate_hs or hashrate_hs <= 0:
        return 0.0
    # Monero: ~2.5 min block time, ~250 KH/s = ~0.003 XMR/day (rough)
    daily_xmr = (hashrate_hs / 1000000.0) * 0.003 * 86400 / 150  # Very rough approximation
    return daily_xmr


def generate_cipher_lines(count: int = 8):
    import random

    syllables = [
        "qa",
        "zu",
        "tri",
        "xel",
        "vor",
        "nek",
        "lum",
        "cy",
        "pha",
        "dra",
        "syn",
        "vek",
        "ora",
        "mir",
        "tal",
        "zen",
        "quo",
        "rax",
        "fid",
        "glu",
    ]
    lines = []
    for _ in range(count):
        words = [random.choice(syllables) for _ in range(random.randint(3, 6))]
        suffix = hex(random.getrandbits(32))[2:]
        lines.append(" ".join(words) + " :: " + suffix)
    return lines


TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>QuantumGuardQC3 Dashboard</title>
  <style>
    :root {
      --bg: #050805;
      --card: #0a120a;
      --text: #c8ffcc;
      --muted: #6af28b;
      --accent: #2adf6c;
      --accent-2: #1ba34d;
    }
    body { font-family: "IBM Plex Mono", Menlo, Consolas, monospace; margin: 24px; background: var(--bg); color: var(--text); }
    h1 { color: var(--accent); letter-spacing: 0.5px; }
    .card { background: var(--card); padding: 16px 20px; border-radius: 10px; margin-bottom: 16px; box-shadow: 0 6px 18px rgba(0,0,0,0.45); border: 1px solid #0f1a0f; }
    .row { display: flex; gap: 16px; flex-wrap: wrap; }
    .label { color: var(--muted); }
    button { padding: 10px 16px; border: 1px solid var(--accent); border-radius: 8px; cursor: pointer; font-weight: 700; background: transparent; color: var(--accent); transition: 0.15s ease; }
    button:hover { background: var(--accent); color: #041204; }
    .stop { border-color: #ff6666; color: #ff8f8f; }
    .stop:hover { background: #ff6666; color: #130303; }
    .muted { color: var(--muted); font-size: 13px; }
    code { background: #0f1a0f; padding: 2px 6px; border-radius: 6px; color: #8bff9c; }
    
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: 12px; }
    .metric { background: #0f1a0f; padding: 12px; border-radius: 8px; border-left: 3px solid var(--accent); }
    .metric-label { color: var(--muted); font-size: 12px; text-transform: uppercase; }
    .metric-value { color: var(--accent); font-size: 20px; font-weight: bold; margin-top: 6px; }
    
    .progress-bar { background: #0f1a0f; border: 1px solid #1a2a1a; border-radius: 4px; height: 24px; overflow: hidden; margin-top: 8px; }
    .progress-fill { background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%); height: 100%; transition: width 0.3s ease; display: flex; align-items: center; justify-content: flex-end; padding-right: 6px; color: #000; font-size: 11px; font-weight: bold; }
    
    .earnings-card { background: #0a1a0a; border: 1px solid #1a3a1a; padding: 16px; border-radius: 8px; margin-top: 12px; }
    .earnings-value { color: #4fffad; font-size: 24px; font-weight: bold; }
  </style>
</head>
<body>
  <h1>QuantumGuardQC3 Dashboard</h1>

  <div class="card">
    <div><span class="label">Monero Wallet:</span> <code>{{ wallet }}</code></div>
    <div class="muted">Monero mainnet (starts with '4'). Real telemetry, CryptoNight mining via XMRig.</div>
  </div>

  <div class="card">
    <div class="label">Live Metrics</div>
    <div class="metric-grid">
      <div class="metric">
        <div class="metric-label">⚡ Hashrate</div>
        <div class="metric-value">{{ hashrate_formatted }}</div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: {{ hashrate_percent }}%;"></div>
        </div>
      </div>
      <div class="metric">
        <div class="metric-label">✓ Shares Accepted</div>
        <div class="metric-value">{{ shares_accepted }}</div>
      </div>
      <div class="metric">
        <div class="metric-label">✗ Shares Rejected</div>
        <div class="metric-value">{{ shares_rejected }}</div>
      </div>
    </div>
    
    <div class="earnings-card">
      <div class="metric-label">💰 Est. Daily Earnings</div>
      <div class="earnings-value">{{ daily_xmr_est }} XMR</div>
      <div class="muted" style="margin-top: 8px;">Rough estimate based on current hashrate. Actual earnings depend on difficulty, pool fees, and luck.</div>
    </div>
  </div>

  <div class="card">
    <div class="label">XMRig Control (best-effort)</div>
    <form method="post" action="{{ url_for('control') }}">
      <input type="text" name="xmrig_path" value="{{ xmrig_path }}" style="background:#0f1a0f; color:#8bff9c; border:1px solid var(--accent); padding:6px; border-radius:6px; width:200px;" />
      <input type="text" name="pool" value="{{ pool }}" style="background:#0f1a0f; color:#8bff9c; border:1px solid var(--accent); padding:6px; border-radius:6px; width:200px;" />
      <input type="text" name="port" value="{{ port }}" style="background:#0f1a0f; color:#8bff9c; border:1px solid var(--accent); padding:6px; border-radius:6px; width:90px;" />
      <input type="text" name="wallet" value="{{ wallet }}" style="background:#0f1a0f; color:#8bff9c; border:1px solid var(--accent); padding:6px; border-radius:6px; width:240px;" />
      <input type="text" name="worker" value="{{ worker }}" style="background:#0f1a0f; color:#8bff9c; border:1px solid var(--accent); padding:6px; border-radius:6px; width:160px;" />
      <input type="text" name="api_url" value="{{ telemetry_url }}" style="background:#0f1a0f; color:#8bff9c; border:1px solid var(--accent); padding:6px; border-radius:6px; width:220px;" />
      <div class="row" style="margin-top:12px;">
        <button name="action" value="start">▶ Start XMRig</button>
        <button class="stop" name="action" value="stop">■ Stop XMRig</button>
      </div>
      <div class="muted" style="margin-top:10px;">Set real values before starting. This UI launches a local process only; no remote control.</div>
    </form>
  </div>

  <div class="card">
    <div class="row" style="align-items:center;">
      <div><span class="label">Telemetry URL:</span></div>
      <form method="post" action="{{ url_for('telemetry') }}">
        <input type="text" name="url" value="{{ telemetry_url }}" style="background:#0f1a0f; color:#8bff9c; border:1px solid var(--accent); padding:6px; border-radius:6px; width:260px;" />
        <button name="action" value="poll">Poll Telemetry</button>
      </form>
    </div>
    <div style="margin-top:10px;">
      <div class="label">Telemetry status:</div>
      <div class="muted">{{ telemetry_status }}</div>
      {% if telemetry_payload %}
      <pre style="background:#0f1a0f; color:#8bff9c; padding:10px; border-radius:8px; max-height:320px; overflow:auto;">{{ telemetry_payload }}</pre>
      {% endif %}
    </div>
  </div>

  <div class="card">
    <div class="label">AI Cipher Chatter</div>
    <div class="muted">Synthetic, for flavor only. Refresh page to regenerate.</div>
    <pre style="background:#0f1a0f; color:#8bff9c; padding:10px; border-radius:8px; max-height:220px; overflow:auto;">{% for line in cipher_lines %}{{ line }}
{% endfor %}</pre>
  </div>

  <div class="card">
    <div class="label">Notes</div>
    <ul>
      <li>Monero-compatible: XMRig handles CryptoNight/RandomX hashing. HTTP API required.</li>
      <li>Wallet format: Monero mainnet starts with '4', stagenet with '8', testnet with '9'.</li>
      <li>Popular Monero pools: supportxmr.com, minexmr.com, moneropools.com</li>
      <li>Configure env vars: MONERO_WALLET, MONERO_POOL, MONERO_POOL_PORT, MONERO_WORKER, XMRIG_PATH, XMRIG_API_URL.</li>
    </ul>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    telem_status = "Idle"
    telem_payload = ""
    hashrate_hs = None
    hashrate_formatted = "-- H/s"
    hashrate_percent = 0
    shares_accepted = 0
    shares_rejected = 0
    daily_xmr_est = "0.000000"
    cipher_lines = generate_cipher_lines()
    
    # Parse telemetry from query args if available
    telem_str = request.args.get("telemetry_payload", "")
    if telem_str:
        try:
            import json
            telem_json = json.loads(telem_str)
            hashrate_hs, shares_accepted, shares_rejected = parse_telemetry(telem_json)
            
            if hashrate_hs is not None and hashrate_hs > 0:
                # Format hashrate (KH/s, MH/s, GH/s)
                if hashrate_hs >= 1e9:
                    hashrate_formatted = f"{hashrate_hs/1e9:.2f} GH/s"
                    hashrate_percent = min(100, (hashrate_hs / 1e9 / 5) * 100)  # Assume 5 GH/s max
                elif hashrate_hs >= 1e6:
                    hashrate_formatted = f"{hashrate_hs/1e6:.2f} MH/s"
                    hashrate_percent = min(100, (hashrate_hs / 1e6 / 1) * 100)  # Assume 1 MH/s max
                else:
                    hashrate_formatted = f"{hashrate_hs/1e3:.2f} KH/s"
                    hashrate_percent = min(100, (hashrate_hs / 1e3 / 100) * 100)  # Assume 100 KH/s max
                
                # Estimate daily earnings
                daily_xmr = estimate_daily_earnings(hashrate_hs)
                daily_xmr_est = f"{daily_xmr:.6f}"
        except Exception:
            pass
    
    return render_template_string(
        TEMPLATE,
        wallet=WALLET_ADDRESS,
        telemetry_url=request.args.get("telemetry_url", DEFAULT_API_URL),
        telemetry_status=request.args.get("telemetry_status", telem_status),
        telemetry_payload=telem_str,
        pool=DEFAULT_POOL,
        port=DEFAULT_PORT,
        worker=DEFAULT_WORKER,
        xmrig_path=DEFAULT_XMRIG_PATH,
        hashrate_formatted=hashrate_formatted,
        hashrate_percent=hashrate_percent,
        shares_accepted=shares_accepted,
        shares_rejected=shares_rejected,
        daily_xmr_est=daily_xmr_est,
        cipher_lines=cipher_lines,
    )


@app.route("/control", methods=["POST"])
def control():
    action = request.form.get("action")
    api_url = request.form.get("api_url", DEFAULT_API_URL)
    pool = request.form.get("pool", DEFAULT_POOL)
    port = int(request.form.get("port", DEFAULT_PORT))
    wallet = request.form.get("wallet", WALLET_ADDRESS)
    worker = request.form.get("worker", DEFAULT_WORKER)
    xmrig_path = request.form.get("xmrig_path", DEFAULT_XMRIG_PATH)
    global XM_PROC
    if action == "start" and start_xmrig:
        ok, proc = start_xmrig(xmrig_path, pool, port, wallet, worker, api_port=int(api_url.split(":")[-1]))
        if ok:
            with XM_LOCK:
                XM_PROC = proc
    elif action == "stop" and stop_process:
        with XM_LOCK:
            if XM_PROC:
                stop_process(XM_PROC)
                XM_PROC = None
    return redirect(url_for("index", telemetry_url=api_url))


@app.route("/telemetry", methods=["POST"])
def telemetry():
    url = request.form.get("url", DEFAULT_API_URL)
    ok, payload = fetch_telemetry(url)
    status = "OK" if ok else f"Error: {payload}"
    
    # Convert payload to JSON string for passing through URL
    if isinstance(payload, dict):
        import json
        pretty = json.dumps(payload)
    else:
        pretty = str(payload) if payload else ""
    
    return redirect(url_for("index", telemetry_url=url, telemetry_status=status, telemetry_payload=pretty))


def main():
    parser = argparse.ArgumentParser(description="QuantumGuardQC3 Dashboard (telemetry)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
