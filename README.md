# Efficient XR Head-Pose Forecasting

Efficient, strictly causal prediction of future head orientation from streaming
quaternion motion. The project studies the accuracy–latency–model-size frontier
for deployable XR motion prediction.

## Task

Given up to 400 past samples at 100 Hz, predict the next 20 unit quaternions.
The primary accuracy measure is mean geodesic angular error at the twentieth
future sample.

```mermaid
flowchart LR
    A[Past unit quaternions] --> B[Sign continuity]
    B --> C[Relative rotations / tangent features]
    C --> D[Causal TCN or state-space encoder]
    D --> E[20-step residual rotation forecast]
    E --> F[Unit-normalized quaternions]
    F --> G[Geodesic error, latency, MACs, parameters]
```

## Data governance

BOXRR-23 contains human motion data and requires acceptance of the Berkeley
Data Use Agreement. It is licensed CC BY-NC-SA 4.0. The DUA prohibits
deanonymization, sensitive-attribute inference and health research. This
project uses the data only for aggregate XR motion forecasting.

Official source and DOI:

- <https://rdi.berkeley.edu/metaverse/boxrr-23/>
- <https://doi.org/10.25350/B5NP4V>

Raw recordings, user identifiers and extracted per-user trajectories are never
committed. Splits are user-disjoint and stored as hashed manifests.

## Evaluation-first roadmap

1. Implement exact quaternion geodesic evaluation and constant-pose/velocity
   baselines.
2. Build a selective BOXRR-23 downloader after explicit DUA acceptance.
3. Filter sessions by sampling rate and motion content without downloading the
   full multi-terabyte corpus.
4. Compare causal MLP, GRU, TCN and compact state-space models.
5. Report user-disjoint confidence intervals, parameters, MACs and CPU latency.
6. Export the selected model to ONNX and test float, dynamic-quantized and
   distilled variants.

