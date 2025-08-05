"""Tests for config.HypofuseRunConfig."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from hypofuse.config import HypofuseRunConfig


def test_from_parts_serializes_configs() -> None:
    """Use lightweight stand-ins for the real configs."""

    @dataclass(frozen=True)
    class FakeNorm:
        case_fold: bool = True

    @dataclass(frozen=True)
    class FakeFusion:
        policy: str = "majority"

    @dataclass(frozen=True)
    class FakeFixture:
        n_utterances: int = 10

    cfg = HypofuseRunConfig.from_parts(
        normalization=FakeNorm(),
        fusion=FakeFusion(),
        fixture=FakeFixture(),
    )
    assert cfg.normalization == {"case_fold": True}
    assert cfg.fusion == {"policy": "majority"}
    assert cfg.fixture == {"n_utterances": 10}


def test_validate_accepts_defaults() -> None:
    cfg = HypofuseRunConfig(
        normalization={},
        fusion={},
        fixture={},
    )
    cfg.validate()


def test_validate_rejects_bad_lm_order() -> None:
    cfg = HypofuseRunConfig(
        normalization={},
        fusion={},
        fixture={},
        lm_order=0,
    )
    with pytest.raises(ValueError, match="lm_order must be >= 1"):
        cfg.validate()


def test_validate_rejects_nonfinite_lm_weight() -> None:
    cfg = HypofuseRunConfig(
        normalization={},
        fusion={},
        fixture={},
        lm_weight=float("inf"),
    )
    with pytest.raises(ValueError, match="lm_weight must be finite"):
        cfg.validate()


def test_validate_rejects_negative_seed() -> None:
    cfg = HypofuseRunConfig(
        normalization={},
        fusion={},
        fixture={},
        seed=-1,
    )
    with pytest.raises(ValueError, match="seed must be >= 0"):
        cfg.validate()


def test_from_parts_with_real_configs() -> None:
    from hypofuse.fixtures import FixtureConfig
    from hypofuse.fusion import FusionConfig
    from hypofuse.normalize import NormalizationConfig

    cfg = HypofuseRunConfig.from_parts(
        normalization=NormalizationConfig(),
        fusion=FusionConfig(),
        fixture=FixtureConfig(),
        lm_order=4,
        lm_weight=0.7,
        seed=42,
    )
    assert cfg.lm_order == 4
    assert cfg.lm_weight == 0.7
    assert cfg.seed == 42
    cfg.validate()
