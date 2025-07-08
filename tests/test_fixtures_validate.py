"""Validation rules for extended FixtureConfig fields."""

from __future__ import annotations

import math

import pytest

from hypofuse.fixtures import FixtureConfig


def test_rate_sum_exceeds_one_raises() -> None:
    with pytest.raises(ValueError, match="rate sum"):
        FixtureConfig(
            substitution_rate=0.5,
            insertion_rate=0.3,
            deletion_rate=0.3,
        ).validate()


def test_rate_sum_exactly_one_passes() -> None:
    FixtureConfig(
        substitution_rate=0.5,
        insertion_rate=0.3,
        deletion_rate=0.2,
    ).validate()


def test_group_bias_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="group_bias key"):
        FixtureConfig(
            speaker_groups=("A", "B"),
            group_bias={"C": 1.0},
        ).validate()


def test_group_bias_negative_raises() -> None:
    with pytest.raises(ValueError, match="group_bias"):
        FixtureConfig(group_bias={"A": -1.0}).validate()


def test_group_bias_non_finite_raises() -> None:
    with pytest.raises(ValueError, match="group_bias"):
        FixtureConfig(group_bias={"A": math.inf}).validate()


def test_rank_decay_negative_raises() -> None:
    with pytest.raises(ValueError, match="rank_decay"):
        FixtureConfig(rank_decay=-0.1).validate()


def test_noise_sensitivity_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="noise_sensitivity"):
        FixtureConfig(noise_sensitivity=11.0).validate()


def test_noise_sensitivity_boundary_passes() -> None:
    FixtureConfig(noise_sensitivity=10.0).validate()
    FixtureConfig(noise_sensitivity=-10.0).validate()
