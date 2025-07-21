"""Declarative configuration loading, hashing, and merging.

Provides generic serialization for frozen dataclasses, JSON I/O, content
hashing, and a top-level HypofuseRunConfig that composes the per-module
configs.
"""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
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
