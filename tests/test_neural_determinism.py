"""Model determinism and prediction bounds."""

from __future__ import annotations

import pytest

from hypofuse.neural import (
    FEATURE_NAMES,
    NeuralConfidenceModel,
    synthetic_confidence_dataset,
)

torch = pytest.importorskip("torch")


@pytest.mark.model
def test_same_seed_identical_predictions() -> None:
    features, labels = synthetic_confidence_dataset(n=100, seed=0)
    m1 = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=42)
    m1.fit(features, labels, epochs=10)
    m2 = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=42)
    m2.fit(features, labels, epochs=10)
    p1 = [round(v, 6) for v in m1.predict_proba(features)]
    p2 = [round(v, 6) for v in m2.predict_proba(features)]
    assert p1 == p2


@pytest.mark.model
def test_predict_proba_in_unit_interval() -> None:
    features, labels = synthetic_confidence_dataset(n=100, seed=0)
    model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=0)
    model.fit(features, labels, epochs=10)
    probs = model.predict_proba(features)
    assert len(probs) == len(features)
    assert all(0.0 <= p <= 1.0 for p in probs)


@pytest.mark.model
def test_different_seeds_differ() -> None:
    features, labels = synthetic_confidence_dataset(n=100, seed=0)
    m1 = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=0)
    m1.fit(features, labels, epochs=10)
    m2 = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=99)
    m2.fit(features, labels, epochs=10)
    p1 = m1.predict_proba(features)
    p2 = m2.predict_proba(features)
    assert p1 != p2
