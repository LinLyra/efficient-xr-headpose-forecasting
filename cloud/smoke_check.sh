#!/usr/bin/env bash
set -euo pipefail

cd /opt/headpose
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/run_synthetic_benchmark.py > /tmp/synthetic-benchmark.json
python - <<'PY'
import json

with open("/tmp/synthetic-benchmark.json", encoding="utf-8") as handle:
    report = json.load(handle)

assert report["steady"]["constant_velocity"]["mean"] < 1e-8
assert report["stationary"]["constant_pose"]["mean"] < 1e-8
print("headpose cloud smoke check passed")
PY
