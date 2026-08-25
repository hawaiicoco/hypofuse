# Usage Workflows

This document provides worked examples for common hypofuse workflows
beyond the quick-start in the README.

## Fitting a temperature calibrator on synthetic scores

This example generates a synthetic calibration dataset, fits a
temperature parameter, and checks the expected calibration error.

```python
from hypofuse.confidence import (
    expected_calibration_error,
    fit_temperature,
    synthetic_calibration_set,
)

# Generate synthetic scores with a known ground-truth relationship.
# This is NOT real ASR data.
scores, labels, p_true = synthetic_calibration_set(n=2000, seed=42)

# Fit temperature by grid search minimising NLL.
best_t = fit_temperature(scores, [float(l) for l in labels])
print(f"best temperature: {best_t}")

# Evaluate ECE on the raw scores (as probabilities in [0, 1]).
ece = expected_calibration_error(
    [max(0.0, min(1.0, s)) for s in scores],
    [float(l) for l in labels],
)
print(f"ECE (uncalibrated): {ece:.4f}")
```

For a simpler approach using the built-in `calibrate` dispatcher:

```python
from hypofuse.confidence import calibrate, calibration_report

scores, labels, _ = synthetic_calibration_set(n=1000, seed=0)
calibrated = calibrate(
    scores,
    [float(l) for l in labels],
    method="temperature",
)
report = calibration_report(calibrated, [float(l) for l in labels])
print(f"ECE: {report['ece']:.4f}, Brier: {report['brier_score']:.4f}")
```

### Fitting a logistic calibrator

```python
from hypofuse.confidence import (
    fit_logistic,
    synthetic_calibration_set,
    brier_score,
)

scores, labels, _ = synthetic_calibration_set(n=1000, seed=7)
cal = fit_logistic(scores, [float(l) for l in labels])
calibrated = cal.apply(scores)
bs = brier_score(calibrated, labels)
print(f"coef={cal.coef:.4f}, intercept={cal.intercept:.4f}")
print(f"Brier={bs:.4f}")
```

## Training the optional neural confidence model

This workflow requires the `torch` extra (CPU only). All data is
synthetic; no real ASR benchmark is implied.

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu "hypofuse[torch]"
```

```python
from hypofuse.neural import (
    FEATURE_NAMES,
    NeuralConfidenceModel,
    baseline_accuracy,
    synthetic_confidence_dataset,
)

# Generate synthetic labelled data.
features, labels = synthetic_confidence_dataset(n=400, seed=0)

# Train the tiny MLP.
model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=0)
losses = model.fit(features, labels, epochs=30, lr=0.05)

# Evaluate.
probs = model.predict_proba(features)
acc = baseline_accuracy(probs, labels)
print(f"epochs={len(losses)}, final_loss={losses[-1]:.4f}")
print(f"acc={acc:.3f}")
```

To run the neural model tests:

```bash
pytest -m model
```
