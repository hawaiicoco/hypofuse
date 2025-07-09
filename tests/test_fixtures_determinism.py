"""Determinism and seed sensitivity tests."""

from __future__ import annotations

import json

from hypofuse.fixtures import (
    FixtureConfig,
    as_manifest_dicts,
    generate_fixture,
)


def _manifest_text(cfg: FixtureConfig) -> str:
    return json.dumps(
        as_manifest_dicts(generate_fixture(cfg), config=cfg),
        sort_keys=True,
    )


def test_same_seed_byte_identical() -> None:
    cfg = FixtureConfig(n_utterances=10, seed=42)
    assert _manifest_text(cfg) == _manifest_text(cfg)


def test_different_seed_different() -> None:
    a = _manifest_text(FixtureConfig(n_utterances=10, seed=42))
    b = _manifest_text(FixtureConfig(n_utterances=10, seed=43))
    assert a != b


def test_config_field_change_alters_output() -> None:
    a = _manifest_text(FixtureConfig(n_utterances=10, seed=42))
    b = _manifest_text(FixtureConfig(n_utterances=10, seed=42, substitution_rate=0.1))
    assert a != b


def test_no_global_random_state_leak() -> None:
    import random

    before = random.getstate()
    generate_fixture(FixtureConfig(n_utterances=5, seed=99))
    after = random.getstate()
    assert before == after
