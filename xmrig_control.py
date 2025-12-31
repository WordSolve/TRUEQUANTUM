"""Control helpers for XMRig HTTP API and local process.

Notes:
- This is best-effort and depends on your XMRig version and API settings.
- The HTTP API typically supports read-only endpoints; control/pause/resume may
  vary by version/build.
- Starting XMRig as a subprocess requires a valid XMRig binary path and config.
"""
import os
import signal
import subprocess
import time
from typing import Optional, Tuple

import requests


def api_get(url: str, timeout: float = 2.0):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        try:
            return True, r.json()
        except Exception:
            return True, r.text
    except Exception as e:
        return False, str(e)


def api_post(url: str, payload: dict, timeout: float = 2.0):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        try:
            return True, r.json()
        except Exception:
            return True, r.text
    except Exception as e:
        return False, str(e)


def pause(api_base: str, timeout: float = 2.0):
    # Not all XMRig builds expose pause/resume; treat failures gracefully.
    return api_post(f"{api_base.rstrip('/')}/control", {"action": "pause"}, timeout=timeout)


def resume(api_base: str, timeout: float = 2.0):
    return api_post(f"{api_base.rstrip('/')}/control", {"action": "resume"}, timeout=timeout)


def submit_share(api_base: str, nonce_hex: str, timeout: float = 2.0):
    # Experimental: some builds expose a submit endpoint; many do not.
    return api_post(f"{api_base.rstrip('/')}/submit", {"nonce": nonce_hex}, timeout=timeout)


def start_xmrig(xmrig_path: str, pool_url: str, pool_port: int, wallet: str, worker: str, api_port: int = 8080) -> Tuple[bool, Optional[subprocess.Popen]]:
    if not os.path.isfile(xmrig_path):
        return False, None
    cmd = [
        xmrig_path,
        "-o", f"{pool_url}:{pool_port}",
        "-u", wallet,
        "-k",
        "--rig-id", worker,
        "--http-host", "0.0.0.0",
        "--http-port", str(api_port),
    ]
    try:
        proc = subprocess.Popen(cmd)
        time.sleep(0.5)
        return True, proc
    except Exception:
        return False, None


def stop_process(proc: subprocess.Popen):
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    except Exception:
        pass
