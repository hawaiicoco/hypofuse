"""Manifest system row and extra nbest metadata."""

from __future__ import annotations

from hypofuse.fixtures import (
    FixtureConfig,
    as_manifest_dicts,
    generate_fixture,
)


def test_system_row_emitted() -> None:
    cfg = FixtureConfig(n_utterances=2, seed=0)
    fx = generate_fixture(cfg)
    rows = as_manifest_dicts(fx, config=cfg)
    system_rows = [r for r in rows if r["schema"] == "hypofuse.system"]
    assert len(system_rows) == 1
    assert system_rows[0]["system_id"] == "synthetic"
    assert system_rows[0]["language"] == "en"


def test_system_row_has_config_hash() -> None:
    cfg = FixtureConfig(n_utterances=2, seed=0)
    fx = generate_fixture(cfg)
    rows = as_manifest_dicts(fx, config=cfg)
    sys_row = next(r for r in rows if r["schema"] == "hypofuse.system")
    # SHA-256 hex digest is 64 characters
    assert len(sys_row["config_hash"]) == 64


def test_nbest_rows_have_extra_metadata() -> None:
    cfg = FixtureConfig(n_utterances=2, seed=0)
    fx = generate_fixture(cfg)
    rows = as_manifest_dicts(fx, config=cfg)
    nbest_rows = [r for r in rows if r["schema"] == "hypofuse.nbest"]
    assert len(nbest_rows) > 0
    for row in nbest_rows:
        assert "speaker_group" in row
        assert "intent_domain" in row
        assert "noise_db" in row
        assert "duration_s" in row


def test_system_row_without_config_has_empty_hash() -> None:
    fx = generate_fixture(FixtureConfig(n_utterances=1, seed=0))
    rows = as_manifest_dicts(fx)
    sys_row = next(r for r in rows if r["schema"] == "hypofuse.system")
    assert sys_row["config_hash"] == ""
