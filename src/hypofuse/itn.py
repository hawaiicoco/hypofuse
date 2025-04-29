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

    def validate(self) -> None:
        """Raise ``ValueError`` for unknown language or all-flags-off."""
        if self.language not in ("en", "zh"):
            raise ValueError(f"unknown language: {self.language!r}")
        if not any([self.numbers, self.decimals, self.percent, self.units, self.negative]):
            raise ValueError("at least one conversion flag must be enabled")

    def with_overrides(self, **changes: object) -> ItnConfig:
        """Return a new config with the given fields replaced."""
        return replace(self, **changes)


_EN_ONES: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_EN_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def en_words_to_int(text: str) -> int:
    """Convert an English number phrase to an integer (0..999999).

    Accepts hyphens ("twenty-one"), the word "and" ("one hundred and
    five"), and scale words "hundred", "thousand", "million". Raises
    ``ValueError`` on empty input or unknown words.
    """
    tokens = text.lower().replace("-", " ").split()
    tokens = [t for t in tokens if t != "and"]
    if not tokens:
        raise ValueError("empty number phrase")
    result = 0
    current = 0
    saw_value = False
    for token in tokens:
        if token in _EN_ONES:
            current += _EN_ONES[token]
            saw_value = True
        elif token in _EN_TENS:
            current += _EN_TENS[token]
            saw_value = True
        elif token == "hundred":
            current *= 100
        elif token == "thousand":
            result += current * 1000
            current = 0
        elif token == "million":
            result += current * 1_000_000
            current = 0
        else:
            raise ValueError(f"unknown number word: {token!r}")
    result += current
    if not saw_value:
        raise ValueError(f"no numeric value found in: {text!r}")
    return result


_ZH_DIGITS: dict[str, int] = {
    "\u96f6": 0,
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
}

_ZH_SMALL_UNITS: dict[str, int] = {
    "\u5341": 10,
    "\u767e": 100,
    "\u5343": 1000,
}


def _parse_zh_section(text: str) -> int:
    """Parse a Chinese number section (up to 9999)."""
    result = 0
    current = 0
    for ch in text:
        if ch in _ZH_DIGITS:
            current = _ZH_DIGITS[ch]
        elif ch in _ZH_SMALL_UNITS:
            unit = _ZH_SMALL_UNITS[ch]
            if current == 0 and unit == 10:
                current = 1
            result += current * unit
            current = 0
        else:
            raise ValueError(f"unknown Chinese digit: {ch!r}")
    return result + current


def zh_words_to_int(text: str) -> int:
    """Convert a Chinese number phrase to an integer (0..99999999).

    Supports: \\u96f6\\u4e00..\\u4e5d, \\u5341\\u767e\\u5343\\u4e07.
    Raises ``ValueError`` on empty input or unknown characters.
    """
    if not text:
        raise ValueError("empty number phrase")
    wan = "\u4e07"
    if wan in text:
        parts = text.split(wan)
        if len(parts) != 2:
            raise ValueError(f"multiple wan markers in: {text!r}")
        wan_part = _parse_zh_section(parts[0]) if parts[0] else 1
        lower_part = _parse_zh_section(parts[1]) if parts[1] else 0
        return wan_part * 10000 + lower_part
    return _parse_zh_section(text)
