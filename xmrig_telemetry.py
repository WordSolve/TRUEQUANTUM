#!/usr/bin/env python3
"""Non-invasive XMRig telemetry script.

Performs a simple GET against the XMRig HTTP API and prints JSON status.
This script is safe and does not submit work or change miner state.
"""
import argparse
import json
import sys

try:
    import requests
except Exception:
    print("requests is required: pip install requests", file=sys.stderr)
    raise


def query_api(url: str, timeout: float = 2.0):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        try:
            data = r.json()
        except Exception:
            data = r.text
        return True, data
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="XMRig telemetry (non-invasive)")
    parser.add_argument("--api-url", default="http://127.0.0.1:8080", help="XMRig HTTP API base URL")
    parser.add_argument("--timeout", type=float, default=2.0, help="Request timeout seconds")
    args = parser.parse_args()

    ok, data = query_api(args.api_url, timeout=args.timeout)
    if ok:
        print("XMRig API reachable. Response:")
        if isinstance(data, (str,)):
            print(data)
        else:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    else:
        print(f"Failed to reach XMRig API at {args.api_url}: {data}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
