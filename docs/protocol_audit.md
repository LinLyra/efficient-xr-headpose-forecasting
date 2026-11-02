# Head-pose forecasting protocol audit

This project adapts the public ABC 2027 task definition into an independent,
open research benchmark. It does not use the unreleased competition test set.

## Task

- Input: 100–400 past unit quaternions in scalar-first `(w, x, y, z)` order.
- Sampling: uniform 100 Hz after spherical linear interpolation.
- Output: exactly 20 future unit quaternions (100 ms horizon).
- Causality: no sample at or beyond the prediction boundary may be used.
- Generalization: all reported splits are user-disjoint; no per-user test-time
  fine-tuning.

## Primary metric

Mean sign-invariant SO(3) geodesic angular error at future sample 20:

`theta = 2 * arccos(abs(dot(q_pred, q_true)))`, reported in degrees.

RMS angle and `1 - abs(dot)` are not equivalent and must not be substituted.

## Efficiency report

Every learned model report must include:

- trainable parameter count;
- multiply-accumulate operations per streaming update;
- batch-one median latency after warm-up on a named CPU and GPU;
- peak memory;
- ONNX export status and operator inventory.

## Required baselines

1. Constant pose.
2. Constant angular velocity.
3. Compact causal MLP over relative rotations.
4. Causal GRU.
5. Dilated temporal convolutional network.

## Data governance

BOXRR-23 is licensed CC BY-NC-SA 4.0 and subject to the Berkeley DUA. The
project prohibits deanonymization, identity inference, sensitive-attribute
inference and health research. Raw `.xror` files, user identifiers and
per-user trajectories are excluded from version control. Split manifests use
salted hashes and aggregate counts only.

