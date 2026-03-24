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

---

## hypofuse.alignment

### `AlignmentCosts` (frozen dataclass)

```python
@dataclass(frozen=True)
class AlignmentCosts:
    substitution: float = 1.0
    insertion: float = 1.0
    deletion: float = 1.0
    token_costs: Mapping[tuple[Hashable, Hashable], float] = field(default_factory=dict)
```

`validate()` raises `ValueError` for negative or non-finite costs.

### `Alignment` (frozen dataclass)

```python
@dataclass(frozen=True)
class Alignment:
    ops: tuple[AlignmentOp, ...]
    score: float
```

Properties: `ref_length`, `hyp_length`, `errors`. Method `to_dict()`.

### `edit_alignment(reference, hypothesis, *, costs=None, band=None) -> Alignment`

Standard DP edit alignment. Tie-break order: substitution > insertion > deletion.
`band` restricts to Sakoe-Chiba band; raises `AlignmentError` when too narrow.
String inputs raise `AlignmentError`.

```python
from hypofuse.alignment import edit_alignment
a = edit_alignment(["a", "b", "c"], ["a", "x", "c"])
assert a.errors == 1
assert a.score == 1.0
```

### Rate functions

```python
word_error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float
character_error_rate(reference: str, hypothesis: str) -> float
symmetric_error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float
corpus_error_rates(pairs, level="word") -> CorpusRates
```

Empty-reference policy: returns 0.0 when both are empty, 1.0 when
hypothesis is non-empty.

### `error_breakdown(alignment) -> ErrorBreakdown`

Returns `substitutions`, `insertions`, `deletions`, `total`,
`ref_length`, `hyp_length`.

### Serialization

```python
alignment_to_markdown(alignment, max_rows=20) -> str
alignment_to_jsonl_row(alignment, utterance_id, system="") -> dict
alignment_from_jsonl_row(row) -> Alignment
```

`alignment_from_jsonl_row` raises `ValueError` for unknown schema or
missing fields.

---

## hypofuse.multi_align

### `TokenGrid` (frozen dataclass)

```python
@dataclass(frozen=True)
class TokenGrid:
    columns: tuple[tuple[Hashable, ...], ...]
    hypotheses: tuple[tuple[Hashable, ...], ...]
    back_pointers: tuple[tuple[tuple[int, str], ...], ...]
    pair_alignments: tuple[Alignment, ...]
```

Properties: `width` (column count), `depth` (hypothesis count).
The gap symbol is `GAP = "*"`.

### `progressive_align(hypotheses, pivot="left", order="given") -> TokenGrid`

Progressively aligns hypotheses left-to-right. `order="sorted"` makes
the result permutation-invariant. Raises `AlignmentError` on empty input
or `ValueError` if any hypothesis contains the gap symbol.

### Helper functions

```python
grid_row(grid, hypothesis_index) -> tuple[Hashable, ...]
grid_coverage(grid) -> list[int]
grid_is_consistent(grid) -> bool
grid_determinism_check(hypotheses) -> bool
```

```python
from hypofuse.multi_align import progressive_align
grid = progressive_align([["a", "b"], ["a", "c"]])
assert grid.width == 2
assert grid.depth == 2
```

---

## hypofuse.fusion

### `FusionConfig` (frozen dataclass)

```python
@dataclass(frozen=True)
class FusionConfig:
    policy: str = "majority"     # "majority" | "score_weighted" | "lm_weighted"
    alpha: float = 1.0
    beta: float = 1.0
    tie_break: str = "lexicographic"  # "lexicographic" | "first"
    null_token: str = "*"
```

`validate()` raises `FusionError` for unknown policy, negative alpha/beta,
or unknown tie-break.

### `FusionResult` (frozen dataclass)

```python
@dataclass(frozen=True)
class FusionResult:
    tokens: tuple[str, ...]
    confidences: tuple[float, ...]
    chosen: tuple[tuple[Hashable, float], ...]
```

### `fuse(grid, scores=None, lm_scores=None, config=None) -> FusionResult`

Fuses hypotheses aligned on `grid`. Missing scores default to 0.
Confidence per token is the fraction of candidates that agree with
the chosen token.

### `fusion_invariants(result, inputs) -> bool`

Returns True when fused token set is a subset of input tokens.

```python
from hypofuse.multi_align import progressive_align
from hypofuse.fusion import fuse, FusionConfig
grid = progressive_align([["a", "b"], ["a", "b"], ["a", "c"]])
result = fuse(grid, config=FusionConfig(policy="majority"))
assert result.tokens == ("a", "b")
```

---

## hypofuse.confusion

### Data classes

```python
@dataclass(frozen=True)
class Arc:
    token: Hashable
    posterior: float

@dataclass(frozen=True)
class ConfusionSlot:
    pivot: str
    arcs: tuple[Arc, ...]

@dataclass(frozen=True)
class ConfusionNetwork:
    slots: tuple[ConfusionSlot, ...]
```

`ConfusionNetwork.one_best()` returns the pivot of each slot.
`validate(tol=1e-6)` checks posterior normalization per slot.

### `build_confusion_network(grid, weights=None, keep_epsilon=False) -> ConfusionNetwork`

Builds a confusion network from an alignment grid. Posteriors are
vote mass normalized. Ties broken lexicographically. When
`keep_epsilon=False` (default), gap arcs are excluded.

### `minimal_cut_one_best(network) -> tuple[str, ...]`

Identical to `network.one_best()` — the per-slot argmax decomposes
the global minimum-cost path.

### Serialization

```python
confusion_to_json(network) -> str
confusion_from_json(payload) -> ConfusionNetwork
confusion_to_manifest_row(network, utterance_id, system="fusion") -> dict
confusion_from_manifest_row(row) -> ConfusionNetwork
```

JSON schema: `"hypofuse.confusion"`, version `1`.

### Comparison helpers

```python
rover_diff(network, rover_tokens) -> list[int]
consistent_with_rover(network, rover_tokens, pivot_tie_break="lexicographic") -> bool
```
