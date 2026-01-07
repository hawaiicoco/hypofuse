"""Text normalization for scoring.

The defaults are tuned for English word-level WER and Chinese character-level
CER. The pipeline is intentionally simple and deterministic so that
re-running it on the same input always produces the same output.

Pipeline (applied in this fixed order -- the order is part of the contract
and is tested):

1. Unicode normalization (configurable: NFC, NFD, NFKC, NFKD).
2. Full-width / half-width unification.
3. Control and variation-selector removal (Cc/Cf plus U+FE00-FE0F and
   U+E0100-E01EF, except ASCII whitespace; configurable).
4. Punctuation stripping (honoring keep_punctuation exemptions).
5. Digit normalization (keep / spoken / hash / digits).
6. Case folding or lowering (configurable mode).
7. Whitespace collapsing.
8. Inverse text normalization (optional, off by default).

The ``digits_to="spoken"`` mode converts digit runs to word form using
:func:`hypofuse.itn.digits_to_words` (for example ``"123"`` becomes
``"one hundred twenty three"``).  This replaced the earlier placeholder
behaviour that substituted every individual digit with ``"0"``.

Normalization is deterministic and offline: no randomness, no network, no
model inference.  Re-running :func:`normalize` on the same input with the
same config always produces the same output.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, fields, replace

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
_DIGIT_RUN_RE = re.compile(r"\d+(?:\.\d+)?")
_WHITESPACE_KEEP = frozenset("\t\n\r ")
_VALID_UNICODE_FORMS = ("NFC", "NFD", "NFKC", "NFKD")
_VALID_CASE_MODES = ("fold", "lower", "none")
_VALID_DIGITS_TO = ("keep", "spoken", "hash", "digits")
_VALID_TOKENIZATION = ("auto", "word", "char", "char_bigram")
_VALID_POLICIES = ("skip", "error", "pass")


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration of the normalization pipeline.

    Field defaults preserve backward compatibility with earlier versions.
    All fields are validated in ``__post_init__``.
    """

    case_fold: bool = True
    strip_punctuation: bool = True
    collapse_whitespace: bool = True
    fullwidth_to_halfwidth: bool = True
    digits_to: str = "keep"
    empty_reference_policy: str = "skip"
    tokenization: str = "auto"
    language_hint: str = ""
    itn: bool = False
    keep_punctuation: tuple[str, ...] = ()
    unicode_form: str = "NFC"
    remove_control: bool = True
    case_mode: str = "fold"

    def __post_init__(self) -> None:
        """Validate field values; raises ValueError on invalid config."""
        if self.unicode_form not in _VALID_UNICODE_FORMS:
            raise ValueError(
                f"unicode_form must be one of {_VALID_UNICODE_FORMS}, got {self.unicode_form!r}"
            )
        if self.case_mode not in _VALID_CASE_MODES:
            raise ValueError(
                f"case_mode must be one of {_VALID_CASE_MODES}, got {self.case_mode!r}"
            )
        if self.digits_to not in _VALID_DIGITS_TO:
            raise ValueError(f"digits_to must be one of {_VALID_DIGITS_TO}, got {self.digits_to!r}")
        if self.tokenization not in _VALID_TOKENIZATION:
            raise ValueError(
                f"tokenization must be one of {_VALID_TOKENIZATION}, got {self.tokenization!r}"
            )
        if self.empty_reference_policy not in _VALID_POLICIES:
            raise ValueError(
                f"empty_reference_policy must be one of"
                f" {_VALID_POLICIES}, got"
                f" {self.empty_reference_policy!r}"
            )

    def with_overrides(self, **changes: object) -> NormalizationConfig:
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)


def _has_cjk(text: str) -> bool:
    """Return True when *text* contains at least one CJK Unified Ideograph."""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _collapse(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


# Variation selectors are category Mn, but they carry presentation hints
# rather than text, so scoring treats them as noise.
_VARIATION_SELECTORS = frozenset(
    chr(code_point) for code_point in (*range(0xFE00, 0xFE10), *range(0xE0100, 0xE01F0))
)


def _remove_control(text: str) -> str:
    """Strip Cc/Cf characters and variation selectors, keeping ASCII space."""
    return "".join(
        ch
        for ch in text
        if ch not in _VARIATION_SELECTORS
        and (unicodedata.category(ch) not in ("Cc", "Cf") or ch in _WHITESPACE_KEEP)
    )


def _strip_punct(text: str, keep: tuple[str, ...]) -> str:
    """Remove punctuation characters, exempting those in *keep*."""
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P") or ch in keep)


def _replace_digits_with_words(text: str, language: str) -> str:
    """Replace each digit run with its word-form representation."""
    from hypofuse.itn import ItnConfig, digits_to_words

    cfg = ItnConfig(language=language or "en")

    def _repl(m: re.Match[str]) -> str:
        return digits_to_words(m.group(), cfg)

    return _DIGIT_RUN_RE.sub(_repl, text)


def _replace_words_with_digits(text: str, language: str) -> str:
    """Replace number-word phrases with digit strings."""
    from hypofuse.itn import ItnConfig, words_to_digits

    cfg = ItnConfig(language=language or "en")
    return words_to_digits(text, cfg)


def normalize(text: str, config: NormalizationConfig | None = None) -> str:
    """Apply the normalization pipeline to *text*.

    An empty input always returns an empty string regardless of policy.
    The ``empty_reference_policy`` is meaningful only at the scorer /
    corpus level, not here.
    """
    config = config or NormalizationConfig()
    if not text:
        return ""
    # 1. Unicode form
    result = unicodedata.normalize(config.unicode_form, text)
    # 2. Full-width folding
    if config.fullwidth_to_halfwidth:
        result = result.translate(_FULLWIDTH_TRANSLATE)
    # 3. Control removal
    if config.remove_control:
        result = _remove_control(result)
    # 4. Punctuation
    if config.strip_punctuation:
        result = _strip_punct(result, config.keep_punctuation)
        # Re-normalize: stripping chars between a base and a combining
        # mark can create new composition opportunities.
        result = unicodedata.normalize(config.unicode_form, result)
    # 5. Digits
    if config.digits_to == "spoken":
        result = _replace_digits_with_words(result, config.language_hint)
    elif config.digits_to == "hash":
        result = _DIGITS_RE.sub("#", result)
    elif config.digits_to == "digits":
        result = _replace_words_with_digits(result, config.language_hint)
    # 6. Case
    if config.case_fold and config.case_mode != "none":
        if config.case_mode == "fold":
            result = result.casefold()
        elif config.case_mode == "lower":
            result = result.lower()
    # 7. Whitespace
    if config.collapse_whitespace:
        result = _collapse(result)
    # Removing a character can leave a base letter next to a combining mark
    # that was blocked from composing earlier, so the chosen Unicode form is
    # applied once more here. This final pass is what makes ``normalize``
    # idempotent: a second run has nothing left to strip or compose.
    result = unicodedata.normalize(config.unicode_form, result)
    # 8. ITN
    if config.itn:
        from hypofuse.itn import ItnConfig, words_to_digits

        itn_cfg = ItnConfig(language=config.language_hint or "en")
        result = words_to_digits(result, itn_cfg)
    return result


def tokenize(text: str, config: NormalizationConfig | None = None) -> tuple[str, ...]:
    """Split normalized text into tokens according to the config.

    Auto picks ``"char"`` when CJK is detected, otherwise ``"word"``.
    ``"char_bigram"`` produces overlapping character bigrams for CJK
    text and falls back to word tokenization for non-CJK text.
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
    if mode == "char_bigram":
        if _has_cjk(normalized):
            chars = normalized.replace(" ", "")
            if len(chars) <= 1:
                return tuple(chars)
            return tuple(chars[i : i + 2] for i in range(len(chars) - 1))
        # Non-CJK: fall back to word tokenization
        return tuple(normalized.split(" "))
    # mode == "word"
    return tuple(normalized.split(" "))


def normalize_pair(
    reference: str,
    hypothesis: str,
    config: NormalizationConfig | None = None,
) -> tuple[str, str]:
    """Normalize a reference and hypothesis with the same config.

    Under the ``"error"`` policy, a :class:`HypofuseError` is raised when
    the reference is empty or whitespace-only.  Under ``"skip"`` and
    ``"pass"`` the pair is returned unchanged (the distinction matters at
    the scorer level, not here).
    """
    config = config or NormalizationConfig()
    if config.empty_reference_policy == "error" and not reference.strip():
        raise HypofuseError("empty reference is not allowed under 'error' policy")
    return normalize(reference, config), normalize(hypothesis, config)


# ------------------------------------------------------------------
# Presets
# ------------------------------------------------------------------

PRESETS: dict[str, NormalizationConfig] = {
    "score_en": NormalizationConfig(tokenization="word"),
    "score_zh": NormalizationConfig(tokenization="char"),
    "raw": NormalizationConfig(
        case_fold=False,
        case_mode="none",
        strip_punctuation=False,
        collapse_whitespace=False,
        fullwidth_to_halfwidth=False,
        remove_control=False,
    ),
    "display": NormalizationConfig(
        strip_punctuation=False,
        case_fold=False,
        case_mode="none",
    ),
}


def preset(name: str) -> NormalizationConfig:
    """Return a named :class:`NormalizationConfig` recipe.

    Raises :class:`ValueError` listing valid names for an unknown one.
    """
    if name not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown preset {name!r}; valid presets: {valid}")
    return PRESETS[name]


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------


def normalization_report(
    reference: str,
    hypothesis: str,
    config: NormalizationConfig | None = None,
) -> dict[str, object]:
    """Build a diagnostic dict for a normalized pair.

    Keys: ``reference``, ``hypothesis``, ``reference_tokens``,
    ``hypothesis_tokens``, ``tokenization_mode``, ``config_hash``.
    """
    config = config or NormalizationConfig()
    ref_norm = normalize(reference, config)
    hyp_norm = normalize(hypothesis, config)
    ref_tok = tokenize(reference, config)
    hyp_tok = tokenize(hypothesis, config)
    mode = config.tokenization
    if mode == "auto":
        mode = "char" if _has_cjk(ref_norm) or config.language_hint.startswith("zh") else "word"
    config_dict: dict[str, object] = {}
    for f in fields(config):
        val = getattr(config, f.name)
        config_dict[f.name] = list(val) if isinstance(val, tuple) else val
    from hypofuse.util import stable_hash

    return {
        "reference": ref_norm,
        "hypothesis": hyp_norm,
        "reference_tokens": ref_tok,
        "hypothesis_tokens": hyp_tok,
        "tokenization_mode": mode,
        "config_hash": stable_hash(config_dict),
    }
