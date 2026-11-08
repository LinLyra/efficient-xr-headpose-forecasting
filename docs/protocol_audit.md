# Head-pose forecasting protocol audit

This project adapts the public ABC 2027 task definition into an independent,
open research benchmark. It does not use the unreleased competition test set.

## Task

- Input: 100–400 past unit quaternions in scalar-first `(w, x, y, z)` order.
- Sampling: uniform 100 Hz after spherical linear interpolation.
- Output: exactly 20 future unit quaternions.
- Protocol ambiguity: the public task simultaneously states 100 Hz input and a
  20-sample/100 ms horizon. At 100 Hz, 20 future intervals span 200 ms. Code
  therefore treats `horizon_samples` and physical duration as separate fields;
  official-style results use step 20 and research results also report errors by
  timestamp.
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

## First cloud pilot (not a final benchmark)

The first end-to-end BOXRR-23 pilot was executed on Runpod without downloading
raw recordings locally. It used three user archives (123,811,840 downloaded
bytes), extracted 133 Beat Saber recordings, and deleted the raw TAR files after
conversion. The processed pilot contains 1,720,251 frames spanning 6.14 hours.
Native median sampling rate was 79.92 Hz; 37 recordings were excluded from the
first evaluation because they contained gaps above 100 ms.

At a 200 ms / 20-sample horizon, 73,331 windows produced the following pilot
means: constant pose 2.88 degrees, last-sample constant velocity 5.99 degrees,
damped velocity 5.14 degrees, and 100 ms windowed velocity 4.87 degrees. The
same ranking held in a provisional high-motion stratum. These are diagnostic
figures from only three users, not publishable estimates.

This result changes the working hypothesis: raw angular-velocity extrapolation
is highly sensitive to tracking jitter, direction changes, and stopping. The
learned component should estimate *whether and how much to extrapolate*, with
constant pose as a protected fallback. The next benchmark must enlarge the
user count, segment gaps instead of dropping whole recordings, compare 100 ms
and 200 ms physical horizons, and report user-clustered uncertainty.

## Expanded cloud benchmark v1

The expanded audit sampled 50 user archives across 16 archive prefixes. It
converted 1,049 recordings (14,937,700 frames, 49.23 hours), segmented 188 gap
boundaries, and deleted every downloaded TAR after conversion. The evaluation
uses a five-second window stride to limit correlation and compute. One selected
user had no evaluable window, so user-cluster bootstrap intervals contain 49
independent user clusters. Selection is size-bounded and prefix-balanced rather
than a probability sample; these are benchmark-v1 estimates, not final BOXRR
population estimates.

At 100 ms, constant pose achieved a 2.29-degree window mean, 5.53-degree p95,
and 2.30-degree user-macro mean (95% user-cluster bootstrap interval
2.16–2.45). Constant velocity was worse by 0.39 degrees per user on average
(paired 95% interval 0.24–0.54); damped velocity was worse by 0.23 degrees
(0.09–0.37).

At 200 ms, constant pose achieved a 3.61-degree window mean, 9.04-degree p95,
and 3.64-degree user-macro mean (95% interval 3.39–3.91). Constant velocity was
worse by 2.08 degrees (paired 95% interval 1.78–2.40); damped velocity was worse
by 1.32 degrees (1.06–1.59). Thus the reversal is statistically stable across
users, not an artifact of a few long recordings.

The next learned baseline is therefore a protected residual/gating model: it
must default to constant pose when extrapolation is unsafe and learn corrections
only for acceleration and sustained turns. Primary model comparisons will use
paired user-level intervals and motion-regime breakdowns, with a held-out
user-disjoint split fixed before tuning.

## Learned benchmark v1

The fixed keyed split assigns 40 users to training, five to validation and five
to testing. A bounded reservoir supplies 120,000/20,000/20,000 windows with no
user overlap. The 17,852-parameter depthwise TCN trained for 20 epochs on one
RTX 4090 and exported successfully to ONNX.

On the untouched test users, the model reduced mean 200 ms error from 3.666 to
3.185 degrees and p95 from 9.096 to 7.637 degrees. The per-user paired change
was -0.477 degrees with a 95% user-cluster bootstrap interval of
[-0.676, -0.302]. The result supports learned selective extrapolation, but the
test cohort remains too small for a final claim.

The final-step gate mean was 0.017 and uncertainty–squared-error correlation was
-0.002. Residual magnitude and gate scale are not separately identifiable in
this first parameterization, so the gate is not yet interpretable. Uncertainty
calibration is also unsuccessful. Both are explicit next-stage problems rather
than claimed contributions of benchmark v1.

## Bounded intervention benchmark v2

Benchmark v2 keeps the split, windows, optimizer and architecture fixed and
changes one modeling assumption: each tangent-space correction component is
bounded to 25 degrees before gating. This removes the arbitrary trade-off
between an unbounded candidate correction and a near-zero gate.

The bounded model achieved a 3.188-degree mean, 7.635-degree p95 and
3.173-degree user-macro mean. Its paired improvement over constant pose was
-0.474 degrees with a 95% user-cluster bootstrap interval of
[-0.669, -0.305]. Accuracy is effectively unchanged from v1, while the mean
final-step gate rises from 0.017 to 0.103 and now has a defined intervention
scale.

The uncertainty head does not predict calibrated error magnitude: its Pearson
correlation with squared error remains -0.002. It does, however, rank difficult
samples, with a final-step rank correlation of 0.506. Mean error at 80%, 50%
and 20% least-uncertain coverage is 2.621, 2.248 and 1.831 degrees,
respectively, versus 3.188 degrees at full coverage. This supports selective
prediction but not probabilistic calibration. The next evaluation must use a
larger held-out cohort and add explicit calibration metrics.

