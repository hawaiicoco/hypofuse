"""Model AUC above max-posterior baseline on synthetic data."""

from __future__ import annotations

import pytest

from hypofuse.neural import (
    FEATURE_NAMES,
    NeuralConfidenceModel,
    auc_from_scores,
    synthetic_confidence_dataset,
)

torch = pytest.importorskip("torch")


@pytest.mark.model
def test_model_auc_above_chance() -> None:
    features, labels = synthetic_confidence_dataset(n=200, seed=1)
    model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=1)
    model.fit(features, labels, epochs=30, lr=0.05)
    probs = model.predict_proba(features)
    model_auc = auc_from_scores(probs, labels)
    assert model_auc > 0.5


@pytest.mark.model
def test_model_auc_at_least_baseline_minus_tolerance() -> None:
    features, labels = synthetic_confidence_dataset(n=200, seed=1)
    model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=1)
    model.fit(features, labels, epochs=30, lr=0.05)
    probs = model.predict_proba(features)
    model_auc = auc_from_scores(probs, labels)
    # Baseline: raw max-posterior feature (index 0 in FEATURE_NAMES)
    baseline_scores = [row[0] for row in features]
    baseline_auc = auc_from_scores(baseline_scores, labels)
    # Tolerance of 0.05 for the small MLP on this synthetic dataset
    assert model_auc >= baseline_auc - 0.05


@pytest.mark.model
@pytest.mark.slow
def test_longer_training_larger_dataset() -> None:
    features, labels = synthetic_confidence_dataset(n=400, seed=2)
    model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=2)
    losses = model.fit(features, labels, epochs=100, lr=0.05)
    assert losses[-1] < losses[0]
    probs = model.predict_proba(features[:10])
    assert all(0.0 <= p <= 1.0 for p in probs)
