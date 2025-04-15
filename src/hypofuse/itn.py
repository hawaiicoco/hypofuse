"""Inverse text normalization: spoken form to written form.

Converts number phrases, decimals, negatives, percents and unit expressions
between spoken (word) and written (digit) representations for English and
Chinese. The :class:`ItnConfig` dataclass controls which conversions are
active and which language is targeted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ItnConfig:
    """Configuration for inverse text normalization.

    Parameters
    ----------
    language:
        Target language, either ``"en"`` or ``"zh"`` (Chinese).
    numbers:
        Convert cardinal number phrases.
    decimals:
        Convert decimal number phrases (e.g. "three point one four").
    percent:
        Convert percent expressions (e.g. "twenty percent").
    units:
        Convert unit expressions (e.g. "kilograms" to "kg").
    negative:
        Convert negative number phrases (e.g. "minus five").
    """

    language: str = "en"
    numbers: bool = True
    decimals: bool = True
    percent: bool = True
    units: bool = True
    negative: bool = True

    def with_overrides(self, **changes: object) -> ItnConfig:
        """Return a new config with the given fields replaced."""
        return replace(self, **changes)
