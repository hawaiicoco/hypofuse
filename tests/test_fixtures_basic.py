"""Deterministic fixture generation."""

from __future__ import annotations

from hypofuse.fixtures import FixtureConfig, generate_fixture


def test_default_fixture_size() -> None:
    fx = generate_fixture(FixtureConfig(n_utterances=5, seed=42))
    assert len(fx) == 5


def test_fixture_has_n_best_hypotheses() -> None:
    fx = generate_fixture(FixtureConfig(n_utterances=2, n_best=4, seed=1))
    for u in fx:
        assert len(u.hypotheses) == 4


def test_fixture_is_deterministic() -> None:
    cfg = FixtureConfig(n_utterances=3, seed=99)
    a = generate_fixture(cfg)
    b = generate_fixture(cfg)
    assert [u.reference for u in a] == [u.reference for u in b]
    assert [u.hypotheses for u in a] == [u.hypotheses for u in b]


def test_fixture_invalid_config_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        FixtureConfig(n_utterances=0).validate()


def test_fixture_speaker_groups_round_trip() -> None:
    fx = generate_fixture(FixtureConfig(n_utterances=10, seed=3))
    groups = {u.speaker_group for u in fx}
    assert groups.issubset({"A", "B", "C"})
