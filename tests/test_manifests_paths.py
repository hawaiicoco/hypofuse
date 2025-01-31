"""Tests for manifest audio path normalization and joining."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import SchemaError
from hypofuse.manifests.paths import normalize_audio_path, safe_audio_join


def test_normalize_audio_path_empty_allowed() -> None:
    assert normalize_audio_path("") == ""


def test_normalize_audio_path_keeps_forward_slash() -> None:
    assert normalize_audio_path("corpus/en/u1.wav") == "corpus/en/u1.wav"


def test_normalize_audio_path_converts_backslash() -> None:
    assert normalize_audio_path("corpus\\en\\u1.wav") == "corpus/en/u1.wav"


def test_normalize_audio_path_rejects_absolute() -> None:
    with pytest.raises(SchemaError, match="must be relative"):
        normalize_audio_path("/etc/passwd")


def test_normalize_audio_path_rejects_tilde() -> None:
    with pytest.raises(SchemaError, match="must be relative"):
        normalize_audio_path("~/u1.wav")


def test_normalize_audio_path_rejects_parent() -> None:
    with pytest.raises(SchemaError, match="parent-directory"):
        normalize_audio_path("a/../b.wav")


def test_normalize_audio_path_rejects_nul() -> None:
    with pytest.raises(SchemaError, match="NUL"):
        normalize_audio_path("a\0b")


def test_normalize_audio_path_rejects_unsafe_chars() -> None:
    with pytest.raises(SchemaError, match="unsafe characters"):
        normalize_audio_path("a/b$c.wav")


def test_safe_audio_join_blocks_traversal(tmp_path) -> None:
    with pytest.raises(SchemaError, match="parent-directory"):
        safe_audio_join(tmp_path, "../outside.wav")


def test_safe_audio_join_allows_relative(tmp_path) -> None:
    target = safe_audio_join(tmp_path, "corpus/u1.wav")
    assert target == (tmp_path / "corpus" / "u1.wav").resolve()
