"""Model training: fit returns loss list and loss decreases on synthetic data."""

from __future__ import annotations

import math

import pytest

from hypofuse.neural import (
    FEATURE_NAMES,
    NeuralConfidenceModel,
    synthetic_confidence_dataset,
)

torch = pytest.importorskip("torch")


@pytest.mark.model
def test_fit_returns_loss_list_of_requested_length() -> None:
    features, labels = synthetic_confidence_dataset(n=100, seed=0)
    model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=0)
    losses = model.fit(features, labels, epochs=10)
    assert len(losses) == 10
    assert all(isinstance(v, float) for v in losses)


@pytest.mark.model
def test_loss_decreases_on_synthetic_dataset() -> None:
    features, labels = synthetic_confidence_dataset(n=200, seed=0)
    model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=0)
    losses = model.fit(features, labels, epochs=30, lr=0.05)
    # On this synthetic dataset the loss decreases from its initial value
    assert losses[-1] < losses[0]


@pytest.mark.model
def test_all_losses_are_finite() -> None:
    features, labels = synthetic_confidence_dataset(n=100, seed=0)
    model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=0)
    losses = model.fit(features, labels, epochs=15)
    assert all(math.isfinite(v) for v in losses)
