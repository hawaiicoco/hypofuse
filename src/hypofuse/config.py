"""Declarative configuration loading, hashing, and merging.

Provides generic serialization for frozen dataclasses, JSON I/O, content
hashing, and a top-level HypofuseRunConfig that composes the per-module
configs.
"""

from __future__ import annotations

import json
import math
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

T = TypeVar("T")


def to_dict(config: Any) -> dict[str, Any]:
    """Recursively convert a frozen dataclass to a plain dict.

    Tuples become lists so the result is JSON-serializable.
    """
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError("to_dict expects a dataclass instance")
    out: dict[str, Any] = {}
    for f in fields(config):
        value = getattr(config, f.name)
        if is_dataclass(value) and not isinstance(value, type):
            out[f.name] = to_dict(value)
        elif isinstance(value, tuple):
            out[f.name] = list(value)
        else:
            out[f.name] = value
    return out


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Reconstruct a frozen dataclass from a plain dict.

    Resolves string annotations via ``typing.get_type_hints`` so nested
    dataclass fields are rebuilt recursively.  Lists are restored to
    tuples when the annotation mentions ``tuple``.
    """
    if not is_dataclass(cls):
        raise TypeError("from_dict expects a dataclass type")
    hints = get_type_hints(cls)
    field_map = {f.name: f for f in fields(cls)}
    unknown = [k for k in data if k not in field_map]
    if unknown:
        raise ValueError(f"unknown keys: {unknown}")
    kwargs: dict[str, Any] = {}
    for name, fld in field_map.items():
        if name not in data:
            if fld.default is MISSING and fld.default_factory is MISSING:
                raise ValueError(f"missing required key: {name}")
            continue
        value = data[name]
        hint = hints.get(name)
        if isinstance(value, list) and "tuple" in str(hint):
            value = tuple(value)
        if isinstance(value, dict) and isinstance(hint, type) and is_dataclass(hint):
            value = from_dict(hint, value)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(cls: type[T], path: Path | str) -> T:
    """Load a frozen dataclass from a JSON file."""
    text = Path(path).read_text(encoding="utf-8")
    return from_dict(cls, json.loads(text))


def dump_config(config: Any, path: Path | str) -> None:
    """Write a frozen dataclass to JSON with indent=2, sorted keys, trailing newline."""
    text = json.dumps(to_dict(config), indent=2, sort_keys=True)
    Path(path).write_text(text + "\n", encoding="utf-8")


def config_hash(config: Any) -> str:
    """Content-based hash via ``stable_hash`` over the canonical dict."""
    from hypofuse.util import stable_hash

    return stable_hash(to_dict(config))


def merge_configs(base: T, overrides: dict[str, Any]) -> T:
    """Return a new frozen instance with dotted-path overrides applied.

    The base is never mutated; nested fields are addressed via
    ``"inner.field"`` dotted paths.
    """
    cls = type(base)
    data = to_dict(base)
    for path, value in overrides.items():
        parts = path.split(".")
        target = data
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise ValueError(f"unknown path: {path}")
            target = target[part]
        if not isinstance(target, dict) or parts[-1] not in target:
            raise ValueError(f"unknown path: {path}")
        target[parts[-1]] = value
    return from_dict(cls, data)


@dataclass(frozen=True)
class HypofuseRunConfig:
    """Top-level configuration composing per-module configs for a run."""

    normalization: dict[str, Any]
    fusion: dict[str, Any]
    fixture: dict[str, Any]
    lm_order: int = 3
    lm_weight: float = 0.5
    seed: int = 0

    def validate(self) -> None:
        """Raise ``ValueError`` on invalid hyper-parameters."""
        if self.lm_order < 1:
            raise ValueError("lm_order must be >= 1")
        if not math.isfinite(self.lm_weight):
            raise ValueError("lm_weight must be finite")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

    @classmethod
    def from_parts(
        cls,
        normalization: Any | None = None,
        fusion: Any | None = None,
        fixture: Any | None = None,
        **kwargs: Any,
    ) -> HypofuseRunConfig:
        """Build from real or stand-in config instances.

        Missing parts default to the project defaults.  Each part is
        serialized with :func:`to_dict` so the result is a plain dict.
        """
        from hypofuse.fixtures import FixtureConfig
        from hypofuse.fusion import FusionConfig
        from hypofuse.normalize import NormalizationConfig

        norm_obj = NormalizationConfig() if normalization is None else normalization
        fusion_obj = FusionConfig() if fusion is None else fusion
        fixture_obj = FixtureConfig() if fixture is None else fixture
        return cls(
            normalization=to_dict(norm_obj),
            fusion=to_dict(fusion_obj),
            fixture=to_dict(fixture_obj),
            **kwargs,
        )
