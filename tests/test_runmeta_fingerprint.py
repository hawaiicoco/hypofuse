"""Tests for runmeta.fingerprint_files."""

from __future__ import annotations

from pathlib import Path

from hypofuse.runmeta import fingerprint_files


def test_fingerprint_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    result = fingerprint_files([empty])
    # SHA-256 of empty bytes (hand-derived)
    assert result[str(empty)] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_fingerprint_ascii_file(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hello world", encoding="utf-8")
    result = fingerprint_files([f])
    assert str(f) in result
    assert len(result[str(f)]) == 64


def test_fingerprint_multiple_files_sorted(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a")
    b.write_text("b")
    result = fingerprint_files([b, a])
    keys = list(result.keys())
    assert keys == sorted(keys)


def test_fingerprint_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "data.txt"
    f.write_text("deterministic")
    r1 = fingerprint_files([f])
    r2 = fingerprint_files([f])
    assert r1 == r2
