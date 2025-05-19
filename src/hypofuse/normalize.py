"""Text normalization for scoring.

The defaults are tuned for English word-level WER and Chinese character-level
CER. The pipeline is intentionally simple and deterministic so that
re-running it on the same input always produces the same output.

Pipeline (configurable, applied in order):

1. Unicode normalization: NFC.
2. Full-width / half-width unification.
3. Punctuation stripping.
4. Case folding (English only by default).
5. Digit normalization (keep / spoken / hash).
6. Whitespace collapsing.
7. Tokenization: whitespace split (English) or per-character (Chinese / mixed).

The :class:`NormalizationConfig` dataclass toggles each step. The
:func:`normalize` entry point applies them in a fixed order.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

from hypofuse.exceptions import HypofuseError

_FULLWIDTH_TRANSLATE = {
    0xFF21: "A",
    0xFF22: "B",
    0xFF23: "C",
    0xFF24: "D",
    0xFF25: "E",
    0xFF26: "F",
    0xFF27: "G",
    0xFF28: "H",
    0xFF29: "I",
    0xFF2A: "J",
    0xFF2B: "K",
    0xFF2C: "L",
    0xFF2D: "M",
    0xFF2E: "N",
    0xFF2F: "O",
    0xFF30: "P",
    0xFF31: "Q",
    0xFF32: "R",
    0xFF33: "S",
    0xFF34: "T",
    0xFF35: "U",
    0xFF36: "V",
    0xFF37: "W",
    0xFF38: "X",
    0xFF39: "Y",
    0xFF3A: "Z",
    0xFF41: "a",
    0xFF42: "b",
    0xFF43: "c",
    0xFF44: "d",
    0xFF45: "e",
    0xFF46: "f",
    0xFF47: "g",
    0xFF48: "h",
    0xFF49: "i",
    0xFF4A: "j",
    0xFF4B: "k",
    0xFF4C: "l",
    0xFF4D: "m",
    0xFF4E: "n",
    0xFF4F: "o",
    0xFF50: "p",
    0xFF51: "q",
    0xFF52: "r",
    0xFF53: "s",
    0xFF54: "t",
    0xFF55: "u",
    0xFF56: "v",
    0xFF57: "w",
    0xFF58: "x",
    0xFF59: "y",
    0xFF5A: "z",
    0xFF10: "0",
    0xFF11: "1",
    0xFF12: "2",
    0xFF13: "3",
    0xFF14: "4",
    0xFF15: "5",
    0xFF16: "6",
    0xFF17: "7",
    0xFF18: "8",
    0xFF19: "9",
}

assert len(_FULLWIDTH_TRANSLATE) == 62

_DIGITS_RE = re.compile(r"\d")


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration of the normalization pipeline."""

    case_fold: bool = True
    strip_punctuation: bool = True
    collapse_whitespace: bool = True
    fullwidth_to_halfwidth: bool = True
    digits_to: str = "keep"  # one of "keep", "spoken", "hash"
    empty_reference_policy: str = "skip"  # "skip" | "error" | "pass"
    tokenization: str = "auto"  # "auto" | "word" | "char"
    language_hint: str = ""
    itn: bool = False  # apply inverse text normalization when True

    def with_overrides(self, **changes: object) -> NormalizationConfig:
        return replace(self, **changes)


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str, config: NormalizationConfig | None = None) -> str:
    """Apply the normalization pipeline to ``text``.

    An empty input always returns an empty string regardless of policy. The
    ``empty_reference_policy`` is meaningful only at the scorer / corpus
    level, not here.
    """
    config = config or NormalizationConfig()
    if not text:
        return ""
    result = unicodedata.normalize("NFC", text)
    if config.fullwidth_to_halfwidth:
        result = result.translate(_FULLWIDTH_TRANSLATE)
    if config.strip_punctuation:
        result = "".join(ch for ch in result if not unicodedata.category(ch).startswith("P"))
    if config.digits_to == "spoken":
        result = _DIGITS_RE.sub("0", result)
    elif config.digits_to == "hash":
        result = _DIGITS_RE.sub("#", result)
    if config.case_fold:
        result = result.casefold()
    if config.collapse_whitespace:
        result = _collapse(result)
    if config.itn:
        # Lazy import keeps ``hypofuse.itn`` free of a normalization dependency.
        from hypofuse.itn import ItnConfig, words_to_digits

        itn_config = ItnConfig(language=config.language_hint or "en")
        result = words_to_digits(result, itn_config)
    return result


def tokenize(text: str, config: NormalizationConfig | None = None) -> tuple[str, ...]:
    """Split normalized text into tokens according to the config.

    Auto picks "char" when CJK is detected, otherwise "word".
    """
    config = config or NormalizationConfig()
    normalized = normalize(text, config)
    if not normalized:
        return ()
    mode = config.tokenization
    if mode == "auto":
        mode = "char" if _has_cjk(normalized) or config.language_hint.startswith("zh") else "word"
    if mode == "char":
        return tuple(normalized.replace(" ", ""))
    return tuple(normalized.split(" "))


def normalize_pair(
    reference: str,
    hypothesis: str,
    config: NormalizationConfig | None = None,
) -> tuple[str, str]:
    """Normalize a reference and hypothesis with the same config."""
    config = config or NormalizationConfig()
    if config.empty_reference_policy == "error" and not reference.strip():
        raise HypofuseError("empty reference is not allowed under 'error' policy")
    return normalize(reference, config), normalize(hypothesis, config)
