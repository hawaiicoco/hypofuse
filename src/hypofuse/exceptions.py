"""Exception hierarchy for hypofuse.

A small set of named errors keeps callers from inspecting strings and lets
upstream CLI code map problems to coherent exit codes.
"""

from __future__ import annotations


class HypofuseError(Exception):
    """Base class for every hypofuse-specific error."""


class SchemaError(HypofuseError):
    """Manifest schema validation failure (missing field, wrong type, ...)."""


class DuplicateIdError(HypofuseError):
    """Two records share an identifier that must be unique."""


class AlignmentError(HypofuseError):
    """Edit alignment received malformed inputs."""


class FusionError(HypofuseError):
    """Consensus fusion received an unsolvable configuration."""


class LanguageModelError(HypofuseError):
    """N-gram language model training or load failure."""


class CalibrationError(HypofuseError):
    """Calibration received an inconsistent set of predictions."""
