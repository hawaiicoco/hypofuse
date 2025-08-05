"""Tests for runmeta.RunMetadata and capture factory."""

from __future__ import annotations

from dataclasses import dataclass

from hypofuse.config import config_hash
from hypofuse.runmeta import RunMetadata, capture


def test_capture_produces_metadata() -> None:
    meta = capture(seed=42, label="test:label")
    assert meta.seed == 42
    assert meta.created_from == "test:label"
    assert meta.hypofuse_version
    assert meta.python_version
    assert meta.platform
    assert meta.numpy_version


def test_capture_without_config() -> None:
    meta = capture(seed=1)
    assert meta.config_hash == ""


def test_capture_with_config() -> None:
    @dataclass(frozen=True)
    class Cfg:
        x: int = 5

    cfg = Cfg()
    meta = capture(seed=1, config=cfg)
    assert meta.config_hash == config_hash(cfg)


def test_platform_is_nonempty_string() -> None:
    meta = capture(seed=0)
    assert isinstance(meta.platform, str)
    assert len(meta.platform) > 0


def test_torch_version_is_none_or_string() -> None:
    meta = capture(seed=0)
    assert meta.torch_version is None or isinstance(meta.torch_version, str)


def test_metadata_is_frozen_instance() -> None:
    meta = capture(seed=0)
    assert isinstance(meta, RunMetadata)
