"""Smoke tests for SystemMetadata."""

from __future__ import annotations

from hypofuse.manifests.system import SystemMetadata


def test_system_metadata_defaults() -> None:
    s = SystemMetadata(system_id="sys_a", language="en")
    assert s.vocabulary_size == 0
    assert s.description == ""
    assert s.acoustic_model == ""
    assert s.language_model == ""


def test_system_metadata_field_assignment_via_constructor() -> None:
    s = SystemMetadata(
        system_id="sys_a",
        language="zh",
        vocabulary_size=9000,
        description="synthetic recognizer",
        acoustic_model="am-1",
        language_model="lm-1",
        decoder="beam",
        version="0.1.0",
    )
    assert s.vocabulary_size == 9000
    assert s.version == "0.1.0"
