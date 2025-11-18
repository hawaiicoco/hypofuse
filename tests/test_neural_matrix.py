"""Feature matrix width invariant."""

from __future__ import annotations

from hypofuse.neural import (
    FEATURE_NAMES,
    confidence_features,
    feature_matrix,
)


def test_feature_names_length() -> None:
    assert len(FEATURE_NAMES) == 8


def test_matrix_width_matches_feature_names() -> None:
    rows = [
        confidence_features([0.25, 0.25, 0.25, 0.25]),
        confidence_features([0.9, 0.1]),
        confidence_features([1.0]),
    ]
    matrix = feature_matrix(rows)
    assert len(matrix) == 3
    for row in matrix:
        assert len(row) == len(FEATURE_NAMES)


def test_matrix_width_exhaustive_shapes() -> None:
    shapes = [
        [1.0],
        [0.5, 0.5],
        [0.33, 0.33, 0.34],
        [0.1, 0.2, 0.3, 0.4],
        [0.05, 0.05, 0.05, 0.05, 0.8],
    ]
    for shape in shapes:
        feat = confidence_features(shape)
        assert len(feat.as_list()) == len(FEATURE_NAMES)


def test_empty_matrix() -> None:
    assert feature_matrix([]) == []
