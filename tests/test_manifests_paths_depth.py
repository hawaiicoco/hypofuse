"""Depth tests for the audio path safety policy."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import SchemaError
from hypofuse.manifests.paths import normalize_audio_path, safe_audio_join


def test_windows_drive_letter_rejected() -> None:
    with pytest.raises(SchemaError, match="drive letter"):
        normalize_audio_path("C:/audio/u1.wav")


def test_windows_drive_letter_backslash() -> None:
    with pytest.raises(SchemaError, match="drive letter"):
        normalize_audio_path("D:\\audio\\u1.wav")


def test_symlink_looking_string_rejected() -> None:
    """Paths containing arrow notation are rejected."""
    with pytest.raises(SchemaError):
        normalize_audio_path("link -> target.wav")


def test_double_dot_only_at_end_rejected() -> None:
    with pytest.raises(SchemaError):
        normalize_audio_path("a/b/..")


def test_empty_string_accepted() -> None:
    assert normalize_audio_path("") == ""


def test_backslash_normalized() -> None:
    assert normalize_audio_path("a\\b\\c.wav") == "a/b/c.wav"


def test_safe_audio_join_absolute_rejected(tmp_path) -> None:
    with pytest.raises(SchemaError, match="must be relative"):
        safe_audio_join(tmp_path, "/etc/passwd")


def test_safe_audio_join_drive_letter(tmp_path) -> None:
    with pytest.raises(SchemaError, match="drive letter"):
        safe_audio_join(tmp_path, "C:/secret.wav")
