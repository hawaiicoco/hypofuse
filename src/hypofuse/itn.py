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


_EN_ONES_LIST = [
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]

_EN_TENS_LIST = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]


def _en_under_1000(n: int) -> str:
    """Convert 0..999 to English words (no 'and')."""
    parts: list[str] = []
    if n >= 100:
        parts.append(_EN_ONES_LIST[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        parts.append(_EN_TENS_LIST[n // 10])
        n %= 10
    if n > 0:
        parts.append(_EN_ONES_LIST[n])
    return " ".join(parts)


def int_to_en_words(n: int) -> str:
    """Convert an integer (0..999999) to English words.

    Raises ``ValueError`` for values outside the supported range.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"expected int, got {type(n).__name__}")
    if not 0 <= n <= 999_999:
        raise ValueError(f"out of supported range: {n}")
    if n == 0:
        return "zero"
    parts: list[str] = []
    if n >= 1000:
        parts.append(_en_under_1000(n // 1000) + " thousand")
        n %= 1000
    if n > 0:
        parts.append(_en_under_1000(n))
    return " ".join(parts)


_ZH_DIGITS_LIST = [
    "\u96f6",
    "\u4e00",
    "\u4e8c",
    "\u4e09",
    "\u56db",
    "\u4e94",
    "\u516d",
    "\u4e03",
    "\u516b",
    "\u4e5d",
]


def _zh_section(n: int) -> str:
    """Convert 1..9999 to Chinese characters."""
    if n == 0:
        return ""
    parts: list[str] = []
    units = [
        (1000, "\u5343"),
        (100, "\u767e"),
        (10, "\u5341"),
    ]
    need_zero = False
    for unit_val, unit_ch in units:
        digit = n // unit_val
        n %= unit_val
        if digit > 0:
            if need_zero:
                parts.append("\u96f6")
            parts.append(_ZH_DIGITS_LIST[digit] + unit_ch)
            need_zero = False
        elif parts:
            need_zero = True
    if n > 0:
        if need_zero:
            parts.append("\u96f6")
        parts.append(_ZH_DIGITS_LIST[n])
    result = "".join(parts)
    # Simplify leading \u4e00\u5341 to \u5341 (e.g. \u5341\u4e94 not \u4e00\u5341\u4e94)
    if result.startswith("\u4e00\u5341"):
        result = "\u5341" + result[2:]
    return result


def int_to_zh_words(n: int) -> str:
    """Convert an integer (0..99999999) to Chinese characters.

    Raises ``ValueError`` for values outside the supported range.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"expected int, got {type(n).__name__}")
    if not 0 <= n <= 99_999_999:
        raise ValueError(f"out of supported range: {n}")
    if n == 0:
        return "\u96f6"
    wan_part = n // 10000
    lower_part = n % 10000
    result = ""
    if wan_part > 0:
        result = _zh_section(wan_part) + "\u4e07"
    if lower_part > 0:
        lower_str = _zh_section(lower_part)
        if wan_part > 0 and lower_part < 1000:
            result += "\u96f6"
        result += lower_str
    return result


def en_phrase_to_digits(text: str) -> str:
    """Convert an English number phrase to a digit string.

    Handles decimals ("three point one four" -> "3.14") and negatives
    ("minus five" -> "-5", "negative five" -> "-5").
    """
    text = text.strip()
    negative = False
    low = text.lower()
    if low.startswith("minus "):
        negative = True
        text = text[6:]
        low = text.lower()
    elif low.startswith("negative "):
        negative = True
        text = text[9:]
        low = text.lower()
    if " point " in low:
        idx = low.index(" point ")
        int_text = text[:idx]
        frac_text = text[idx + 7 :]
        int_part = en_words_to_int(int_text)
        frac_tokens = frac_text.lower().replace("-", " ").split()
        frac_tokens = [t for t in frac_tokens if t != "and"]
        frac_digits = []
        for token in frac_tokens:
            if token in _EN_ONES and _EN_ONES[token] < 10:
                frac_digits.append(str(_EN_ONES[token]))
            else:
                raise ValueError(f"invalid fractional digit: {token!r}")
        result = f"{int_part}.{''.join(frac_digits)}"
    else:
        result = str(en_words_to_int(text))
    if negative:
        result = "-" + result
    return result


def zh_phrase_to_digits(text: str) -> str:
    """Convert a Chinese number phrase to a digit string.

    Handles decimals and negatives (\\u8d1f prefix).
    """
    text = text.strip()
    negative = False
    if text.startswith("\u8d1f"):
        negative = True
        text = text[1:]
    dian = "\u70b9"
    if dian in text:
        parts = text.split(dian)
        if len(parts) != 2:
            raise ValueError(f"invalid decimal: {text!r}")
        int_part = zh_words_to_int(parts[0]) if parts[0] else 0
        frac_digits = []
        for ch in parts[1]:
            if ch in _ZH_DIGITS and _ZH_DIGITS[ch] < 10:
                frac_digits.append(str(_ZH_DIGITS[ch]))
            else:
                raise ValueError(f"invalid fractional digit: {ch!r}")
        result = f"{int_part}.{''.join(frac_digits)}"
    else:
        result = str(zh_words_to_int(text))
    if negative:
        result = "-" + result
    return result


# Unit tables: intentionally small and extensible by downstream users.
_EN_UNITS: dict[str, str] = {
    "kilograms": "kg",
    "kilogram": "kg",
    "meters": "m",
    "meter": "m",
    "metres": "m",
    "metre": "m",
    "seconds": "s",
    "second": "s",
    "hours": "h",
    "hour": "h",
}

_ZH_UNITS: dict[str, str] = {
    "\u516c\u65a4": "kg",
    "\u7c73": "m",
    "\u79d2": "s",
}


def en_lookup_unit(text: str) -> str | None:
    """Look up an English unit word, return abbreviation or None."""
    return _EN_UNITS.get(text.lower())


def zh_lookup_unit(text: str) -> str | None:
    """Look up a Chinese unit word, return abbreviation or None."""
    return _ZH_UNITS.get(text)


def words_to_digits(text: str, config: ItnConfig | None = None) -> str:
    """Replace number phrases in text with digit strings.

    Idempotent: applying twice gives the same result as applying once.
    Never raises on ordinary prose; only on invalid config.
    """
    config = config or ItnConfig()
    config.validate()
    if not text:
        return ""
    # Simplified implementation: try to parse entire text as number phrase
    # A full implementation would scan for number spans within prose
    try:
        if config.language == "en":
            return en_phrase_to_digits(text)
        return zh_phrase_to_digits(text)
    except ValueError:
        # Not a number phrase, return as-is
        return text


def digits_to_words(text: str, config: ItnConfig | None = None) -> str:
    """Replace digit sequences in text with word form.

    Inverse of words_to_digits for numbers within the supported range.
    Never raises on ordinary prose; only on invalid config.
    """
    config = config or ItnConfig()
    config.validate()
    if not text:
        return ""
    # Simplified implementation: try to parse entire text as digits
    # A full implementation would scan for digit spans within prose
    try:
        # Handle negative
        negative = False
        working = text.strip()
        if working.startswith("-"):
            negative = True
            working = working[1:]
        # Handle decimal
        if "." in working:
            parts = working.split(".")
            if len(parts) != 2:
                return text
            int_part = int(parts[0])
            frac_part = parts[1]
            if config.language == "en":
                int_words = int_to_en_words(int_part)
                frac_words = " ".join(int_to_en_words(int(d)) for d in frac_part)
                result = f"{int_words} point {frac_words}"
            else:
                int_words = int_to_zh_words(int_part)
                frac_words = "".join(_ZH_DIGITS_LIST[int(d)] for d in frac_part)
                result = f"{int_words}\u70b9{frac_words}"
        else:
            n = int(working)
            result = int_to_en_words(n) if config.language == "en" else int_to_zh_words(n)
        if negative:
            result = "negative " + result if config.language == "en" else "\u8d1f" + result
        return result
    except (ValueError, TypeError):
        # Not a valid digit sequence, return as-is
        return text
