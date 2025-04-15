"""Fixture manifest conversion."""

from __future__ import annotations

from hypofuse.fixtures import FixtureConfig, as_manifest_dicts, generate_fixture


def test_manifest_dicts_include_nbest_and_reference() -> None:
    fx = generate_fixture(FixtureConfig(n_utterances=2, seed=0))
    rows = as_manifest_dicts(fx)
    schemas = [r["schema"] for r in rows]
    assert "hypofuse.nbest" in schemas
    assert "hypofuse.reference" in schemas


def test_manifest_dicts_preserve_utterance_id() -> None:
    fx = generate_fixture(FixtureConfig(n_utterances=2, seed=0))
    rows = as_manifest_dicts(fx)
    ids = [r["utterance_id"] for r in rows]
    assert ids.count("u0000") == 2  # one nbest + one reference


def test_manifest_dicts_have_deterministic_text() -> None:
    cfg = FixtureConfig(n_utterances=2, seed=7)
    a = as_manifest_dicts(generate_fixture(cfg))
    b = as_manifest_dicts(generate_fixture(cfg))
    assert a == b


def test_manifest_dicts_reference_carries_metadata() -> None:
    fx = generate_fixture(FixtureConfig(n_utterances=1, seed=0))
    rows = as_manifest_dicts(fx)
    ref = next(r for r in rows if r["schema"] == "hypofuse.reference")
    assert "speaker_group" in ref
    assert "duration_s" in ref
    assert "noise_db" in ref
