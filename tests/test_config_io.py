"""Tests for config.load_config and config.dump_config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from hypofuse.config import dump_config, load_config


@dataclass(frozen=True)
class SampleConfig:
    alpha: float
    beta: int
    name: str = "default"


def test_dump_and_load_roundtrip(tmp_path: Path) -> None:
    cfg = SampleConfig(alpha=1.5, beta=42, name="test")
    path = tmp_path / "config.json"
    dump_config(cfg, path)
    restored = load_config(SampleConfig, path)
    assert restored == cfg


def test_dump_produces_stable_format(tmp_path: Path) -> None:
    cfg = SampleConfig(alpha=1.5, beta=42)
    path = tmp_path / "config.json"
    dump_config(cfg, path)
    content1 = path.read_text()
    dump_config(cfg, path)
    content2 = path.read_text()
    assert content1 == content2
    assert content1.endswith("\n")


def test_load_rejects_extra_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"alpha": 1.0, "beta": 2, "name": "x", "extra": "bad"}\n')
    with pytest.raises(ValueError, match=r"unknown keys.*extra"):
        load_config(SampleConfig, path)


def test_roundtrip_byte_identical(tmp_path: Path) -> None:
    cfg = SampleConfig(alpha=2.5, beta=99, name="byte")
    path = tmp_path / "config.json"
    dump_config(cfg, path)
    first_dump = path.read_bytes()
    restored = load_config(SampleConfig, path)
    dump_config(restored, path)
    second_dump = path.read_bytes()
    assert first_dump == second_dump
