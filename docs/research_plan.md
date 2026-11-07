# Research execution plan

## Scope

This is a portfolio-grade, publication-disciplined study rather than a claim of
laboratory-scale foundation-model training. The central hypothesis is that a
small learned residual should intervene mainly during non-stationary motion,
while a calibrated physical predictor should remain the default elsewhere.

## Stage gates

1. **Protocol gate:** geometry, streaming causality, synthetic trajectories and
   official-style metrics pass deterministic tests.
2. **Data gate:** a cloud-only pilot produces valid timestamped quaternion
   shards, a user-disjoint manifest and a data-quality report. No raw recording
   is retained after successful conversion.
3. **Baseline gate:** constant pose, damped velocity, acceleration, regression
   and filtering baselines are evaluated before learned models.
4. **Research gate:** the learned residual must improve high-dynamic regimes
   without materially degrading stationary and steady regimes.
5. **Deployment gate:** the selected model exports to ONNX, uses the permitted
   operator subset and reports batch-one steady-state latency and MACs.

## Compute budget

- Persistent processed data: target 20–50 GB, never raw multi-terabyte storage.
- Pilot raw download: 20–50 GB to ephemeral cloud storage.
- Expansion requires a documented learning-curve or coverage justification.
- Single-GPU training; target total GPU use 100–200 hours.
- Stop-work review at US$100 and hard budget review before US$200.

## First learned model

The initial candidate is a depthwise causal TCN operating on eight invariant
motion features: angular increment, angular acceleration, and their norms. It
predicts a 20-step tangent-space residual, diagonal log-variance and a blend
gate. The residual head starts at zero and the gate starts mostly closed, so an
untrained model preserves the physical predictor instead of emitting arbitrary
rotations. The initial parameter ceiling is 100,000.

Benchmark-v1 trained a 17,852-parameter instance on 120,000 windows from 40
users, selected on 20,000 validation windows from five users and evaluated once
on 20,000 windows from five held-out users. It improved the held-out mean from
3.666 to 3.185 degrees. The user-paired bootstrap interval excludes zero, but
five test users are insufficient for a final population claim.

The next revision must constrain the residual/gate parameterization so the
gate has a physically interpretable intervention magnitude, then calibrate
uncertainty with coverage and risk–coverage curves. The current
uncertainty–error correlation is effectively zero.

## Data boundary

BOXRR-23 access is cloud-only and governed by the applicable Berkeley terms.
Code must never embed credentials. Raw data, stable user identifiers, per-user
trajectories, checkpoints and ONNX exports are excluded from Git.
