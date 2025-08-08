"""Tests for runmeta.embed and runmeta.verify."""

from __future__ import annotations

from hypofuse.runmeta import capture, embed, verify


def test_embed_adds_run_key() -> None:
    meta = capture(seed=1, label="embed")
    rows = [{"id": 1}, {"id": 2}]
    result = embed(rows, meta)
    assert len(result) == 2
    assert "run" in result[0]
    assert "run" in result[1]


def test_embed_does_not_mutate_input() -> None:
    meta = capture(seed=1)
    rows = [{"id": 1}, {"id": 2}]
    original_rows = [dict(r) for r in rows]
    embed(rows, meta)
    assert rows == original_rows
    assert "run" not in rows[0]


def test_embed_preserves_original_fields() -> None:
    meta = capture(seed=1)
    rows = [{"id": 1, "value": "a"}]
    result = embed(rows, meta)
    assert result[0]["id"] == 1
    assert result[0]["value"] == "a"


def test_verify_returns_true_for_matching() -> None:
    meta = capture(seed=5, label="verify")
    rows = [{"id": 1}, {"id": 2}]
    embedded = embed(rows, meta)
    assert verify(embedded, meta) is True


def test_verify_returns_false_for_stale() -> None:
    meta1 = capture(seed=5, label="first")
    meta2 = capture(seed=6, label="second")
    rows = [{"id": 1}]
    embedded = embed(rows, meta1)
    assert verify(embedded, meta2) is False


def test_verify_returns_false_when_run_missing() -> None:
    meta = capture(seed=1)
    rows = [{"id": 1}]
    assert verify(rows, meta) is False


def test_verify_returns_true_for_empty_rows() -> None:
    meta = capture(seed=1)
    assert verify([], meta) is True
