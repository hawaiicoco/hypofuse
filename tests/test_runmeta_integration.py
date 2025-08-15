"""Integration test: build config, capture metadata, embed, verify."""

from __future__ import annotations

from hypofuse.config import HypofuseRunConfig, config_hash
from hypofuse.fixtures import FixtureConfig
from hypofuse.fusion import FusionConfig
from hypofuse.normalize import NormalizationConfig
from hypofuse.runmeta import capture, embed, verify


def test_full_workflow() -> None:
    """Build a HypofuseRunConfig, capture, embed into rows, verify."""
    cfg = HypofuseRunConfig.from_parts(
        normalization=NormalizationConfig(),
        fusion=FusionConfig(),
        fixture=FixtureConfig(),
        lm_order=3,
        lm_weight=0.5,
        seed=123,
    )
    cfg.validate()
    meta = capture(seed=123, config=cfg, label="integration:test")
    assert meta.seed == 123
    assert meta.config_hash == config_hash(cfg)
    assert meta.created_from == "integration:test"
    rows = [
        {"utterance_id": "u0001", "wer": 0.05},
        {"utterance_id": "u0002", "wer": 0.08},
        {"utterance_id": "u0003", "wer": 0.03},
    ]
    embedded = embed(rows, meta)
    assert len(embedded) == 3
    assert verify(embedded, meta)
    cfg2 = HypofuseRunConfig.from_parts(
        normalization=NormalizationConfig(),
        fusion=FusionConfig(),
        fixture=FixtureConfig(),
        lm_order=3,
        lm_weight=0.7,
        seed=123,
    )
    meta2 = capture(seed=123, config=cfg2, label="integration:test")
    assert meta2.config_hash != meta.config_hash
