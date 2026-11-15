# Efficient uncertainty-aware head-motion forecasting on SO(3)

## Research question

Can a compact causal model improve 200 ms head-orientation forecasts by
learning when to depart from a constant-pose fallback, without sacrificing
edge deployment efficiency?

## Protocol

The benchmark contains 50 BOXRR-23 users and 49.23 hours of processed motion.
Users are separated into 40 training, five validation and five test users. Each
run uses 120,000/20,000/20,000 windows. Predictions are tangent-space rotations
relative to the last observed pose, and the primary metric is sign-invariant
SO(3) geodesic error at future sample 20 (200 ms at 100 Hz).

## Main result

At a fixed 50-sample context, all compact learned models improve over constant
pose. The 15,212-parameter GRU is best, reducing mean error from 3.666 to 3.162
degrees. Its paired user-level change is -0.499 degrees with a 95% clustered
bootstrap interval of [-0.693, -0.320].

| Model | Parameters | Mean | Median | p95 |
| --- | ---: | ---: | ---: | ---: |
| Constant pose | 0 | 3.666° | — | 9.096° |
| MLP | 37,900 | 3.198° | 2.524° | 7.679° |
| TCN | 17,852 | 3.188° | 2.517° | 7.656° |
| GRU | 15,212 | **3.162°** | **2.492°** | **7.555°** |

The gain is concentrated where a learned correction is most useful. For the
GRU, the mean change relative to constant pose is -0.149 degrees in low-motion
windows, -0.411 in medium-motion windows and -1.235 in high-motion windows.
This supports the central hypothesis: the protected fallback is strong during
stable motion, while learning matters most during rapid changes.

## Context and uncertainty

The TCN changes little as context grows from 10 to 100 samples: mean error is
3.224, 3.240, 3.188 and 3.183 degrees at contexts 10, 25, 50 and 100. These
runs share test users but not identical reservoir-sampled windows, so the
comparison is evidence of saturation rather than a precise paired effect.

The GRU uncertainty score ranks error better than simple motion heuristics:
rank correlation is 0.521, versus 0.415 for angular speed and 0.284 for angular
acceleration. Retaining the 50% least-uncertain forecasts reduces mean error
from 3.162 to 2.202 degrees. Validation-fitted variance scaling yields a mean
absolute interval-coverage error of 0.028 on the test users.

## Deployment result

The context-50 TCN exports to ONNX. On the cloud Pod CPU with one runtime
thread, FP32 inference has 0.112 ms median and 0.208 ms p99 latency. Dynamic
INT8 reduces file size from 84,512 to 57,082 bytes but increases median latency
to 0.326 ms. For a network this small, quantization overhead dominates; FP32 is
the better operating point on this CPU.

## Scope of the claim

This is a controlled portfolio benchmark, not a population estimate. The test
cohort has only five users, BOXRR-23 covers a narrow application domain, and
latency has not been measured on a phone or XR headset. The strongest result is
therefore empirical: a very small protected residual model improves rapid
head-motion forecasts, long context has limited value, and quantization is not
automatically beneficial at this scale.
