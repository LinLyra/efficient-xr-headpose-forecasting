# Frozen evaluation protocol for event-aware head-motion forecasting

**Protocol version:** 1.0  
**Status:** frozen before the expanded benchmark  
**Freeze date:** 2026-11-17  
**Primary task:** causal 200 ms head-orientation forecasting at 100 Hz

This document fixes the hypotheses, cohorts, metrics, tuning rules and decision
gates before the expanded BOXRR-23 experiment. Changes made after inspecting
test results must be recorded in the amendment table and reported as
exploratory.

## 1. Research question

When does a learned predictor add value beyond a tuned physical predictor, and
can an explicit intervention policy reduce harmful extrapolation or executed
compute without sacrificing endpoint accuracy?

The work tests a principle rather than a particular architecture: physical
prediction is the protected default; learning is justified only for motion
events in which it provides a reproducible benefit.

## 2. Confirmatory hypotheses

- **H1 — event-specific accuracy.** At 200 ms, the selected learned residual
  reduces mean SO(3) error relative to the best validation-selected physical
  baseline in the frozen high-dynamic event set.
- **H2 — protected stability.** The learned residual is non-inferior to that
  physical baseline in stable motion, with a margin of 0.10 degrees in mean
  endpoint error.
- **H3 — explicit intervention.** Relative to an otherwise identical ungated
  residual model, explicit gating reduces harmful interventions and p95 error
  while remaining non-inferior in overall mean error by the same 0.10-degree
  margin.
- **H4 — useful risk ranking.** Learned uncertainty ranks endpoint failures
  better than angular speed and angular acceleration alone.

H1 and H2 determine whether the project continues as a method paper. H3
determines whether explicit gating is retained as a claimed contribution. H4
is secondary and cannot rescue a failure of H1.

## 3. Cohort and immutable split

The target BOXRR cohort contains 250 eligible users:

| Partition | Users | Purpose |
| --- | ---: | --- |
| Training | 160 | fitting models and event thresholds |
| Validation | 40 | all model and baseline selection |
| Test | 50 | one final confirmatory evaluation |

Eligibility is determined before model fitting: valid head rotations, at least
ten minutes of usable motion after gap segmentation, and sufficient samples in
the primary task. If fewer than 250 users satisfy these criteria, the test set
remains 50 users and training is reduced first. No user may cross a partition.

Assignment uses a versioned keyed hash and is stratified, where metadata are
available and reliable, by application, archive prefix and device family. The
public manifest contains salted user hashes, partition, aggregate duration,
application and data-quality fields, but no stable source identifier.

The test reservoir is generated once with a recorded seed. Every method uses
identical test window indices. Each user contributes at most the same
pre-specified number of windows per event stratum so long recordings do not
dominate. Windows that overlap in time are not split across tuning and
evaluation roles.

## 4. Signal processing and task definition

- Quaternions use scalar-first `(w, x, y, z)` order and adjacent signs are made
  continuous before interpolation.
- Recordings are segmented at timestamp gaps greater than 100 ms; interpolation
  never crosses a gap.
- Segments are resampled to 100 Hz by spherical interpolation.
- Input poses and targets are expressed relative to the final observed pose.
- The confirmatory horizon is sample 20, which is 200 ms at 100 Hz.
- The 100 ms endpoint and full 20-step trajectory are secondary outcomes.
- All filtering, feature extraction and prediction are strictly causal.

## 5. Frozen event taxonomy

For consecutive poses `q[t-1], q[t]`, define body-frame angular velocity

`omega[t] = Log(q[t-1]^-1 q[t]) / dt`,

then `alpha[t] = (omega[t] - omega[t-1]) / dt` and
`jerk[t] = (alpha[t] - alpha[t-1]) / dt`, with `dt = 0.01 s`.

Thresholds are estimated from training users only and frozen before validation
or test evaluation:

- `S_low`: 25th percentile of recent RMS `|omega|`;
- `S_high`: 75th percentile of recent RMS `|omega|`;
- `A_high`: 75th percentile of recent RMS `|alpha|`;
- `J_high`: 75th percentile of recent RMS `|jerk|`.

Recent RMS statistics use the final 100 ms of context. Each window receives one
exclusive label in this order:

1. **reversal:** the last two non-trivial velocity vectors have cosine below
   -0.5 and both exceed `S_low`;
2. **braking:** speed falls by at least 50% within the final 100 ms, with peak
   speed above `S_high`;
3. **acceleration/turn onset:** acceleration exceeds `A_high` or jerk exceeds
   `J_high`;
4. **sustained fast turn:** speed exceeds `S_high` without meeting 1–3;
5. **steady:** speed is at or below `S_low` and acceleration is below `A_high`;
6. **other.**

The confirmatory **high-dynamic set** is the union of labels 1–4. Thresholds
will not be chosen to maximize model separation. Event counts and the fraction
of users represented in every event are reported; a label with fewer than 20
test users is exploratory.

## 6. Baselines and equal tuning budget

All non-learned methods are tuned on validation users, using the same endpoint
metric and at most 30 hyperparameter configurations per method:

1. constant pose;
2. last-sample constant angular velocity;
3. windowed robust angular velocity;
4. exponentially damped angular velocity;
5. clipped constant angular acceleration;
6. causal linear velocity trend;
7. SO(3) error-state Kalman filter with constant-velocity dynamics.

Search spaces and the selected configuration are serialized before test
evaluation. The best physical baseline is selected by validation user-macro
mean at 200 ms, not test performance. Learned models receive a matched
validation budget. Required learned controls are a direct residual model, an
explicitly gated residual model, and the protected physical baseline alone.

## 7. Models, seeds and selection

MLP, TCN and GRU encoders are development candidates. Architecture selection
uses validation users only. The selected architecture and its ungated/gated
controls are each trained with seeds `1701, 1702, 1703, 1704, 1705`.

No architecture, feature, event threshold or loss is changed after test results
are opened. Failed runs are reported with their failure reason and are not
silently replaced. Model comparisons report each seed, mean and standard
deviation; inferential comparisons first average each user's error across the
five seeds and then resample users.

## 8. Outcomes

The primary metric is sign-invariant SO(3) geodesic error at 200 ms:

`2 arccos(|<q_pred, q_true>|)`, in degrees.

Primary summaries are user-macro mean over the high-dynamic set and over all
windows. Secondary outcomes are:

- median, p95 and p99 endpoint error;
- 100 ms endpoint and per-step trajectory error;
- error in each frozen event label;
- harmful-intervention rate: fraction of windows where the learned method is
  worse than its protected physical fallback by more than 0.5 degrees;
- excess-intervention severity: mean positive value of
  `error_learned - error_fallback`;
- uncertainty risk-coverage curve and area under that curve;
- AUROC for detecting the worst 20% endpoint errors;
- parameters, serialized size, peak memory, MACs and batch-one p50/p95/p99
  latency on named hardware.

Prediction gating alone is not counted as compute saving because the encoder
still runs. A compute claim requires a separately measured conditional-execution
policy whose cheap trigger skips the neural encoder.

## 9. Statistical analysis

- The independent resampling unit is the user.
- Paired candidate-minus-reference differences use 10,000 user-cluster
  bootstrap resamples and two-sided 95% percentile intervals.
- Confirmatory p-values use 10,000 paired user-level sign-flip permutations;
  non-inferiority tests use the corresponding pre-specified margin.
- Model comparisons use identical windows and are paired within user.
- H1, H2 and H3 form one confirmatory family; p-values are adjusted by Holm's
  procedure. Effect sizes and intervals remain primary.
- H4 comparisons bootstrap per-user differences in risk-ranking metrics.
- Both user-macro and window-micro results are reported; conclusions follow the
  user-macro analysis.
- Event analyses failing the 20-user support rule are exploratory.

## 10. Pre-committed decision gates

### Gate A — method-paper viability

Continue the event-aware method claim only if all conditions hold:

1. in high-dynamic events, the selected learned model improves over the best
   physical baseline by at least 0.25 degrees **and** 5% relative;
2. the 95% paired user interval excludes zero in favor of the learned model
   and the Holm-adjusted confirmatory p-value is below 0.05;
3. stable-motion degradation is no greater than 0.10 degrees;
4. the high-dynamic effect has the same direction in at least four of five
   seeds.

If Gate A fails, development switches to the empirical benchmark/negative-result
track. No new event threshold or subgroup will be introduced to recover H1.

### Gate B — retain explicit gating

Keep explicit gating as a contribution only if overall mean error is
non-inferior within 0.10 degrees and it achieves at least one of:

- a 10% relative reduction in harmful-intervention rate;
- a 5% relative reduction in p95 error;
- a measured 30% reduction in executed neural inference through a real
  conditional-execution policy.

Otherwise the final model is simplified to an ungated residual predictor and
gating is reported as a negative ablation.

### Gate C — uncertainty claim

Retain a learned-uncertainty claim only if its per-user area under the
risk-coverage curve is better than both `|omega|` and `|alpha|`, and the 95%
paired user interval for each comparison excludes zero. Otherwise uncertainty
is reported descriptively, not as a contribution.

## 11. External validity and device evaluation

External data are not pooled with BOXRR. A candidate dataset must provide a
compatible HMD orientation stream, timestamps, licensing for evaluation and
enough duration for a user-disjoint test. All preprocessing adaptations are
declared before evaluating the selected model. Cross-dataset evaluation is a
separate result and does not alter Gates A–C.

Device claims require measurements on an identified phone, XR headset or edge
computer. Desktop/cloud CPU latency is a reproducibility check, not evidence of
mobile latency or energy efficiency. Human-participant data collection or
perceptual evaluation begins only after the applicable ethics review.

## 12. Reproducibility artifacts

The release will contain code, configuration, hashed split manifest, event
thresholds, baseline search spaces, seeds, aggregate per-user metrics, model
cards and environment details. Raw recordings, stable identifiers, per-user
trajectories and credentials remain private under the dataset terms.

## Amendments after freezing

| Date | Change | Reason | Confirmatory impact |
| --- | --- | --- | --- |
| — | None | — | — |
