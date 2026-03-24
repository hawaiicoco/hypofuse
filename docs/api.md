# API Reference

This document covers every public symbol in the `hypofuse` package.
Each entry shows the signature, argument meanings, return type, raised
exceptions, and a short usage example.

## hypofuse.manifests

Re-exports the dataclasses below. Module-level constants:

| Constant | Value |
|---|---|
| `SCHEMA_VERSION` | `1` |
| `SCHEMA_NBEST` | `"hypofuse.nbest"` |
| `SCHEMA_REFERENCE` | `"hypofuse.reference"` |
| `SCHEMA_SYSTEM` | `"hypofuse.system"` |
| `SCHEMA_FUSION_RUN` | `"hypofuse.fusion_run"` |
| `SCHEMA_REPORT` | `"hypofuse.report"` |
| `KNOWN_SCHEMAS` | frozenset of the five schema strings above |

### `NBestHypothesis` (frozen dataclass)

```python
@dataclass(frozen=True)
class NBestHypothesis:
    rank: int
    text: str
    tokens: tuple[str, ...] = ()
    posteriors: tuple[float, ...] = ()
    acoustic_log10: float = 0.0
    lm_log10: float = 0.0
    start_time: float | None = None
    end_time: float | None = None
```

- `rank` — 1-based position in the n-best list.
- `tokens` — positional token strings (words for English, characters for Chinese).
- `acoustic_log10` / `lm_log10` — log-base-10 scores.

```python
from hypofuse.manifests import NBestHypothesis
h = NBestHypothesis(rank=1, text="hello world", tokens=("hello", "world"))
assert h.acoustic_log10 == 0.0
```

### `NBestList` (frozen dataclass)

```python
@dataclass(frozen=True)
class NBestList:
    utterance_id: str
    system: str
    language: str
    hypotheses: tuple[NBestHypothesis, ...] = ()
    audio_path: str = ""
```

`ranks()` returns the tuple of rank values.

### `ReferenceTranscript` (frozen dataclass)

```python
@dataclass(frozen=True)
class ReferenceTranscript:
    utterance_id: str
    text: str
    speaker_id: str = ""
    speaker_group: str = ""
    duration_s: float = 0.0
    noise_db: float = 0.0
    intent_domain: str = ""
    tokens: tuple[str, ...] = ()
```

### `SystemMetadata` (frozen dataclass)

```python
@dataclass(frozen=True)
class SystemMetadata:
    system_id: str
    language: str
    vocabulary_size: int = 0
    description: str = ""
    acoustic_model: str = ""
    language_model: str = ""
    decoder: str = ""
    version: str = ""
```

### `FusionArc` / `FusionRun` (frozen dataclasses)

See the fusion reference section below.

### JSONL helpers (`manifests.jsonl`)

```python
read_manifest(path: Path | str) -> list[dict[str, Any]]
write_manifest(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None
```

- `read_manifest` validates every row and rejects duplicate `utterance_id` values.
  Raises `SchemaError` (with path and 1-based row index) or `DuplicateIdError`.
- `write_manifest` validates each row before writing. Raises `SchemaError`.

### Path safety (`manifests.paths`)

```python
normalize_audio_path(raw: str) -> str
safe_audio_join(base: Path | str, raw: str) -> Path
```

Rejects absolute paths, NUL bytes, parent-directory escapes (`..`),
double slashes, and unsafe characters in segments. Raises `SchemaError`.

### Validation (`manifests.validate`)

```python
validate_record(row: Mapping[str, Any]) -> str
reject_duplicate_ids(rows: Iterable[Mapping[str, Any]], id_field: str = "utterance_id") -> None
```

`validate_record` returns the schema name or raises `SchemaError`.
`reject_duplicate_ids` raises `DuplicateIdError`.

---

## hypofuse.normalize

### `NormalizationConfig` (frozen dataclass)

```python
@dataclass(frozen=True)
class NormalizationConfig:
    case_fold: bool = True
    strip_punctuation: bool = True
    collapse_whitespace: bool = True
    fullwidth_to_halfwidth: bool = True
    digits_to: str = "keep"          # "keep" | "spoken" | "hash"
    empty_reference_policy: str = "skip"  # "skip" | "error" | "pass"
    tokenization: str = "auto"       # "auto" | "word" | "char"
    language_hint: str = ""
    itn: bool = False
```

`with_overrides(**changes)` returns a new instance.

### `normalize(text, config=None) -> str`

Applies NFC normalization, fullwidth unification, punctuation stripping,
case folding, digit replacement, whitespace collapsing, and optional ITN.
Empty input returns `""`.

### `tokenize(text, config=None) -> tuple[str, ...]`

Splits normalized text. `"auto"` picks character-level tokenization when
CJK characters are detected, word-level otherwise.

### `normalize_pair(reference, hypothesis, config=None) -> tuple[str, str]`

Normalizes both strings with the same config. Raises `HypofuseError` when
`empty_reference_policy="error"` and the reference is blank.

```python
from hypofuse.normalize import normalize, NormalizationConfig
assert normalize("Hello, World!") == "hello world"
cfg = NormalizationConfig(case_fold=False)
assert normalize("Hello", cfg) == "Hello"
```

---

## hypofuse.itn

### `ItnConfig` (frozen dataclass)

```python
@dataclass(frozen=True)
class ItnConfig:
    language: str = "en"     # "en" | "zh"
    numbers: bool = True
    decimals: bool = True
    percent: bool = True
    units: bool = True
    negative: bool = True
```

`validate()` raises `ValueError` for unknown language or all flags off.

### Conversion functions

| Function | Signature | Range |
|---|---|---|
| `en_words_to_int` | `(text: str) -> int` | 0..999,999 |
| `zh_words_to_int` | `(text: str) -> int` | 0..99,999,999 |
| `int_to_en_words` | `(n: int) -> str` | 0..999,999 |
| `int_to_zh_words` | `(n: int) -> str` | 0..99,999,999 |
| `en_phrase_to_digits` | `(text: str) -> str` | decimals, negatives |
| `zh_phrase_to_digits` | `(text: str) -> str` | decimals, negatives |
| `words_to_digits` | `(text: str, config=None) -> str` | idempotent |
| `digits_to_words` | `(text: str, config=None) -> str` | inverse |

```python
from hypofuse.itn import en_words_to_int, int_to_en_words
assert en_words_to_int("twenty one") == 21
assert int_to_en_words(325) == "three hundred twenty five"
```
