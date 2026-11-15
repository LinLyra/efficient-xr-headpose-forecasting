"""Measure batch-one ONNX latency and dynamic INT8 size/latency trade-offs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


def latency(session: ort.InferenceSession, values: np.ndarray) -> dict[str, float]:
    input_name = session.get_inputs()[0].name
    for _ in range(100):
        session.run(None, {input_name: values})
    timings = []
    for _ in range(1000):
        started = time.perf_counter_ns()
        session.run(None, {input_name: values})
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "p50_ms": float(np.quantile(timings, 0.5)),
        "p95_ms": float(np.quantile(timings, 0.95)),
        "p99_ms": float(np.quantile(timings, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quantized = args.model.with_name(f"{args.model.stem}-int8.onnx")
    quantize_dynamic(args.model, quantized, weight_type=QuantType.QInt8)
    values = np.zeros((1, args.context - 2, 8), dtype=np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    report = {}
    for name, path in (("float32", args.model), ("dynamic_int8", quantized)):
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        report[name] = {
            "bytes": path.stat().st_size,
            **latency(session, values),
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
