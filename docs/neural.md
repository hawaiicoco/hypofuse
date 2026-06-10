# Neural Confidence Model (optional)

The `hypofuse.neural` module demonstrates the package's confidence
features end to end using a small MLP trained on synthetic posteriors.
It requires the optional `torch` extra and operates on synthetic data
only.

## Installation

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu "hypofuse[torch]"
```

The torch dependency is CPU-only. No GPU is required or used.

## Feature vector

The fixed-width feature vector is defined by `FEATURE_NAMES`:

| Index | Name | Description |
|---|---|---|
| 0 | `max_posterior` | Highest posterior in the distribution |
| 1 | `mean_posterior` | Arithmetic mean of posteriors |
| 2 | `entropy` | Shannon entropy in nats |
| 3 | `posterior_margin` | `top1 - top2` of sorted posteriors |
| 4 | `agreement_ratio` | Vote agreement or max posterior fallback |
| 5 | `n_tokens` | Number of candidates |
| 6 | `acoustic_log10` | Acoustic log10 score |
| 7 | `lm_log10` | Language model log10 score |

## Synthetic dataset

`synthetic_confidence_dataset(n=400, seed=0, error_rate=0.2)`
generates labelled confidence data. Correct tokens (label 1) draw
posteriors from Beta(6, 2) (mean 0.75); incorrect tokens (label 0)
draw from Beta(2, 6) (mean 0.25). The remaining mass is split over
two rivals. This is a controlled synthetic generator with a
constructed relationship; it does not model real ASR output.

## Model

`NeuralConfidenceModel(input_dim, hidden=8, seed=0)` is a tiny MLP:
input -> Linear -> tanh -> Linear -> sigmoid. Training is
deterministic: both `torch.manual_seed` and Python `random.seed` are
set from `seed`.

```python
from hypofuse.neural import (
    NeuralConfidenceModel,
    synthetic_confidence_dataset,
    baseline_accuracy,
    FEATURE_NAMES,
)

features, labels = synthetic_confidence_dataset(n=400, seed=0)
model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=0)
losses = model.fit(features, labels, epochs=30, lr=0.05)
probs = model.predict_proba(features)
acc = baseline_accuracy(probs, labels)
print(f"final loss={losses[-1]:.4f}, accuracy={acc:.3f}")
```

## Running the tests

Tests requiring torch are marked with `@pytest.mark.model`:

```bash
pytest -m model
```

## Honest limitations

- No pretrained weights are shipped or downloaded.
- No real ASR result is claimed; all data is synthetic.
- The model is a demonstration of the confidence pipeline, not a
  production confidence estimator.
- Downstream users should evaluate on their own labelled data
  before drawing conclusions.
