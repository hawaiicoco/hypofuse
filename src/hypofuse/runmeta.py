"""Reproducible run metadata capture and embedding.

Captures version information, platform, seed, and config hash into a frozen
dataclass.  Provides serialization, deterministic stamping, and embedding
into report rows for provenance tracking.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from hypofuse._version import __version__


def _normalize_platform() -> str:
    """Return a short normalized platform string from ``sys.platform``."""
    plat = sys.platform
    if plat.startswith("linux"):
        return "linux"
    if plat == "darwin":
        return "darwin"
    if plat.startswith("win"):
        return "windows"
    return plat


def _torch_version() -> str | None:
    """Return torch version if importable, else None."""
    try:
        import torch  # type: ignore[import-untyped]

        return torch.__version__
    except ImportError:
        return None


@dataclass(frozen=True)
class RunMetadata:
    """Immutable snapshot of run-time versions and configuration."""

    hypofuse_version: str
    python_version: str
    platform: str
    numpy_version: str
    seed: int
    config_hash: str
    created_from: str = ""
    torch_version: str | None = None


def capture(seed: int, config: Any | None = None, label: str = "") -> RunMetadata:
    """Factory to capture run metadata.

    If *config* is provided its hash is computed via :func:`config_hash`.
    Otherwise ``config_hash`` is set to an empty string.
    """
    cfg_hash = ""
    if config is not None:
        from hypofuse.config import config_hash

        cfg_hash = config_hash(config)
    pyver = ".".join(str(v) for v in sys.version_info[:3])
    return RunMetadata(
        hypofuse_version=__version__,
        python_version=pyver,
        platform=_normalize_platform(),
        numpy_version=np.__version__,
        seed=seed,
        config_hash=cfg_hash,
        created_from=label,
        torch_version=_torch_version(),
    )


def to_dict(meta: RunMetadata) -> dict[str, Any]:
    """Convert RunMetadata to a plain dict."""
    return asdict(meta)


def from_dict(data: dict[str, Any]) -> RunMetadata:
    """Reconstruct RunMetadata from a dict."""
    return RunMetadata(**data)


def to_json(meta: RunMetadata) -> str:
    """Serialize RunMetadata to JSON with sorted keys."""
    return json.dumps(to_dict(meta), sort_keys=True, indent=2, ensure_ascii=False)


def from_json(text: str) -> RunMetadata:
    """Deserialize RunMetadata from JSON."""
    return from_dict(json.loads(text))
