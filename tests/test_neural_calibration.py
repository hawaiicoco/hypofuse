"""ECE with model probabilities and temperature scaling bridge."""

from __future__ import annotations

import math

import pytest

from hypofuse.confidence import expected_calibration_error
from hypofuse.neural import (
    FEATURE_NAMES,
    NeuralConfidenceModel,
    calibrate_with_model,
    fit_temperature,
    logistic_temperature_scale,
    synthetic_confidence_dataset,
)

torch = pytest.importorskip("torch")


@pytest.mark.model
def test_calibrate_returns_correct_length() -> None:
    features, labels = synthetic_confidence_dataset(n=100, seed=3)
    model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=3)
    model.fit(features, labels, epochs=10)
    probs = calibrate_with_model(model, features)
    assert len(probs) == len(features)
    assert all(0.0 <= p <= 1.0 for p in probs)


@pytest.mark.model
def test_ece_finite_and_in_unit_interval() -> None:
    features, labels = synthetic_confidence_dataset(n=200, seed=4)
    model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=4)
    model.fit(features, labels, epochs=20)
    probs = calibrate_with_model(model, features)
    accuracies = [float(lbl) for lbl in labels]
    ece = expected_calibration_error(probs, accuracies)
    assert math.isfinite(ece)
    assert 0.0 <= ece <= 1.0


@pytest.mark.model
def test_temperature_does_not_increase_ece() -> None:
    features, labels = synthetic_confidence_dataset(n=200, seed=5)
    model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), seed=5)
    model.fit(features, labels, epochs=20)
    probs = calibrate_with_model(model, features)
    accuracies = [float(lbl) for lbl in labels]
    ece_before = expected_calibration_error(probs, accuracies)
    t = fit_temperature(probs, labels)
    scaled = logistic_temperature_scale(probs, temperature=t)
    ece_after = expected_calibration_error(scaled, accuracies)
    # fit_temperature includes 1.0 (identity) in the default grid,
    # so the result is always at least as good as no scaling.
    assert ece_after <= ece_before + 1e-9
