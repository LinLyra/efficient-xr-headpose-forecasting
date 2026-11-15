"""Poll final-suite artifacts through Jupyter without opening extra terminals."""

from __future__ import annotations

import argparse
import os
import time

from jupyter_bootstrap_client import authenticate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--interval", type=int, default=45)
    parser.add_argument("--timeout", type=int, default=14_400)
    args = parser.parse_args()
    password = os.environ.get("JUPYTER_PASSWORD")
    if not password:
        raise RuntimeError("JUPYTER_PASSWORD is required")
    session = authenticate(args.url.rstrip("/"), password)
    api = f"{args.url.rstrip('/')}/api/contents/workspace/headpose-artifacts/final-suite"
    deadline = time.monotonic() + args.timeout
    previous = None
    while time.monotonic() < deadline:
        response = session.get(api, timeout=30)
        response.raise_for_status()
        entries = response.json().get("content", [])
        names = sorted(entry["name"] for entry in entries)
        snapshot = ", ".join(names)
        if snapshot != previous:
            print(f"artifacts: {snapshot}", flush=True)
            previous = snapshot
        status = session.get(f"{api}/status", timeout=30)
        if status.status_code == 200:
            value = status.json().get("content", "").strip()
            print(f"FINAL_STATUS={value}", flush=True)
            raise SystemExit(0 if value == "0" else 1)
        time.sleep(args.interval)
    raise TimeoutError("final suite exceeded monitor deadline")


if __name__ == "__main__":
    main()
