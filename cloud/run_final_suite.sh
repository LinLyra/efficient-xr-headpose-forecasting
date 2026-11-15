#!/usr/bin/env bash
set -uo pipefail

ROOT=/workspace/headpose-data/boxrr-benchmark-v1
PROJECT=/workspace/xr-headpose-forecasting
OUT=/workspace/headpose-artifacts/final-suite
mkdir -p "$OUT"
cd "$PROJECT" || exit 2

python -m pip install -q -e '.[ml]'
if [[ $? -ne 0 ]]; then
  echo 2 > "$OUT/status"
  exit 2
fi

run_one() {
  local model="$1"
  local context="$2"
  local target="$OUT/${model}-c${context}"
  mkdir -p "$target"
  python scripts/train_boxrr_gate.py "$ROOT" \
    --output-dir "$target" \
    --model "$model" \
    --context "$context" \
    --max-correction-degrees 25 \
    > "$target/train.log" 2>&1
}

overall=0
for context in 10 25 50 100; do
  run_one tcn "$context" || overall=1
  [[ $overall -eq 0 ]] || break
done

if [[ $overall -eq 0 ]]; then
  run_one mlp 50 || overall=1
fi
if [[ $overall -eq 0 ]]; then
  run_one gru 50 || overall=1
fi
if [[ $overall -eq 0 ]]; then
  python scripts/benchmark_onnx.py \
    "$OUT/tcn-c50/residual-gated-tcn.onnx" \
    --context 50 \
    --output "$OUT/tcn-c50/onnx-latency.json" \
    > "$OUT/tcn-c50/latency.log" 2>&1 || overall=1
fi

python - "$OUT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = {}
for report in sorted(root.glob("*-c*/report.json")):
    values = json.loads(report.read_text())
    summary[report.parent.name] = {
        "elapsed_seconds": values["elapsed_seconds"],
        "parameters": values["parameters"],
        "test": values["test"],
    }
latency = root / "tcn-c50" / "onnx-latency.json"
if latency.exists():
    summary["onnx_latency"] = json.loads(latency.read_text())
(root / "suite-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

echo "$overall" > "$OUT/status"
echo "FINAL_SUITE_DONE status=$overall"
exit "$overall"
