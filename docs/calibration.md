# Confidence and Calibration

## Token confidence

`token_confidence_from_posteriors(posteriors)` returns the arithmetic
mean of per-token posterior probabilities. This is a simple summary
statistic, not a calibrated probability.

```python
from hypofuse.confidence import token_confidence_from_posteriors

c = token_confidence_from_posteriors([0.4, 0.6, 0.8])
# c == 0.6
```

## Utterance confidence

`utterance_confidence_from_vote(votes, chosen)` computes per-token
agreement: for each position in `chosen`, it counts how many vote
streams agree, divided by the number of streams that cover that
position.

This is the same quantity that `fuse()` reports as per-token
confidence when using majority voting.

## Temperature scaling

`temperature_scale(scores, temperature=1.0)` applies softmax with
temperature:

```
p_i = exp(s_i / T) / sum_j exp(s_j / T)
```

| Temperature | Effect |
|---|---|
| `T = 1.0` | Standard softmax, distribution unchanged |
| `T > 1.0` | Flattens distribution (more uniform) |
| `T < 1.0` | Sharpens distribution (more peaked) |
| `T -> 0` | Approaches argmax |

Raises `ValueError` for `temperature <= 0`.

### Rank preservation

Temperature scaling preserves the argmax: the index of the maximum
score is always the index of the maximum output probability, for any
positive temperature. This means re-ranking by temperature-scaled
scores produces the same ordering as re-ranking by raw scores.

### Grid search for temperature

To find a good temperature, evaluate ECE on a held-out set across
a grid of values:

```python
from hypofuse.confidence import temperature_scale, expected_calibration_error

best_t, best_ece = 1.0, float("inf")
for t in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    scaled = temperature_scale(val_scores, temperature=t)
    ece = expected_calibration_error(scaled, val_accuracies)
    if ece < best_ece:
        best_t, best_ece = t, ece
```

## Piecewise calibrator

`piecewise_calibrate(scores, breakpoints, slopes, intercepts)` applies
a deterministic piecewise-linear transform. Each segment is defined
by a breakpoint (lower edge), slope, and intercept: `out = slope * s + intercept`.

This transform is monotone when all slopes are positive, preserving
the rank ordering of the input scores.

## ECE and reliability bins

`expected_calibration_error(confidences, accuracies, n_bins=10)`
computes:

```
ECE = sum_b (|avg_conf_b - avg_acc_b| * count_b) / total
```

Bin assignment uses uniform-width bins on [0, 1]:
`bin_index = min(n_bins - 1, max(0, int(confidence * n_bins)))`.

For `n_bins=10`: bin 0 is [0.0, 0.1), bin 1 is [0.1, 0.2), ...,
bin 9 is [0.9, 1.0].

`reliability_bins` returns `CalibrationBin` objects with `lower`,
`upper`, `count`, `avg_confidence`, `avg_accuracy` per bin.

## What calibration on synthetic data shows

The calibration module operates on synthetic score distributions
with known ground truth. It demonstrates that:
- Temperature scaling changes the sharpness of score distributions.
- ECE quantifies the gap between confidence and accuracy.
- The piecewise calibrator can adjust scores deterministically.

It does **not** claim state-of-the-art calibration on real ASR
hypotheses. Users should evaluate calibration on their own labeled
data to draw conclusions about real-world performance.
