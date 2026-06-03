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

### Presets

| Name | Purpose |
|---|---|
| `score_en` | English word-level scoring |
| `score_zh` | Chinese character-level scoring |
| `raw` | Minimal processing |
| `display` | Keep punctuation and case |

`preset(name)` returns a named config.

### Additional NormalizationConfig fields

| Field | Default | Description |
|---|---|---|
| `keep_punctuation` | `()` | Exempt chars |
| `unicode_form` | `NFC` | NFC/NFD/NFKC/NFKD |
| `remove_control` | `True` | Strip Cc/Cf |
| `case_mode` | `fold` | fold/lower/none |

### `digits_to` modes

| Mode | Behaviour |
|---|---|
| `keep` | Leave digits unchanged |
| `spoken` | Digit runs to number words |
| `hash` | Each digit to `#` |
| `digits` | Number words to digit strings |

### Pipeline order (fixed contract)

1. Unicode normalization
2. Full-width unification
3. Control removal
4. Punctuation stripping
5. Digit normalization
6. Case folding
7. Whitespace collapsing
8. ITN (optional)

A final Unicode pass ensures idempotence.

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

### Percent and units

`en_phrase_to_digits` handles negatives ("minus five" -> "-5").
Unit lookup: `en_lookup_unit`, `zh_lookup_unit`.

### Roundtrip stability

`words_to_digits` is idempotent.
`itn_roundtrip_stable(text, config)` verifies stability.

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

---

## hypofuse.ngram

Constants: `UNK = "<unk>"`, `BOS = "<s>"`, `EOS = "</s>"`.

### `NgramLM` (dataclass)

```python
@dataclass
class NgramLM:
    order: int
    vocab: frozenset[str]
    counts: dict[int, Counter]
    total_unigrams: int
```

#### `NgramLM.train(sentences, order=3) -> NgramLM`

Builds an n-gram model from tokenized sentences. Raises
`LanguageModelError` when `order < 1`.

#### `prob_katz(ngram: tuple[str, ...]) -> float`

Katz backoff probability. OOV tokens are mapped to `<unk>`.
Raises `LanguageModelError` for out-of-range n-gram order.

#### `prob_jelinek(ngram, lambdas=None) -> float`

Jelinek-Mercer interpolated probability. Default lambdas are
uniform (`1/order` each). Raises `LanguageModelError` when
lambdas do not sum to 1.

#### `perplexity(tokens, method="katz") -> float`

Geometric mean of inverse probabilities over the padded sequence.
Returns 1.0 for empty input.

#### `to_arpa() -> str` / `NgramLM.from_arpa(payload) -> str`

Export/import in ARPA format (`\data\`, `\N-grams:`, `\end\`).
The roundtrip preserves vocabulary and n-gram counts.

```python
from hypofuse.ngram import NgramLM
lm = NgramLM.train([["the", "cat", "sat"]], order=2)
assert "<unk>" in lm.vocab
p = lm.prob_katz(("the", "cat"))
assert p > 0.0
```

---

## hypofuse.rescore

### `ScoredHypothesis` (frozen dataclass)

```python
@dataclass(frozen=True)
class ScoredHypothesis:
    text: str
    tokens: tuple[str, ...]
    acoustic_log10: float
    lm_log10: float
```

`ScoredHypothesis.from_tokens(tokens, acoustic_log10=0.0)` is a convenience
constructor.

### Functions

```python
rescore_lm(hyp: ScoredHypothesis, lm: NgramLM) -> float
```

Computes LM log10 score using Katz backoff over the hypothesis tokens
padded with `<s>` and `</s>`.

```python
shallow_fusion_score(hyp, lm, lm_weight, acoustic_weight=1.0) -> float
```

Returns `acoustic_weight * acoustic_log10 + lm_weight * lm_score`.

```python
rescore_nbest(nbest, lm, lm_weight, acoustic_weight=1.0) -> tuple[ScoredHypothesis, ...]
```

Re-ranks by combined score descending.

```python
lm_weight_sweep(nbest, reference, lm, weights=(0.0, 0.1, 0.2, 0.5, 1.0))
    -> list[tuple[float, float, str]]
```

Returns `(weight, error_rate, chosen_text)` per weight value.

---

## hypofuse.confidence

### `CalibrationBin` (frozen dataclass)

```python
@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    avg_confidence: float
    avg_accuracy: float
```

### Functions

```python
expected_calibration_error(confidences, accuracies, n_bins=10) -> float
```

Uniform-width bins on [0, 1]. Raises `ValueError` for misaligned
inputs or `n_bins < 1`.

```python
reliability_bins(confidences, accuracies, n_bins=10) -> list[CalibrationBin]
```

Bin assignment: `idx = min(n_bins - 1, max(0, int(c * n_bins)))`.

```python
temperature_scale(scores: Sequence[float], temperature: float = 1.0) -> list[float]
```

Softmax with temperature. Raises `ValueError` for `temperature <= 0`.

```python
piecewise_calibrate(scores, breakpoints, slopes, intercepts) -> list[float]
```

Deterministic piecewise-linear transform. Raises `ValueError` when
parameter lengths do not match.

```python
token_confidence_from_posteriors(posteriors: Sequence[float]) -> float
```

Mean of per-token posteriors. Returns 0.0 for empty input.

```python
utterance_confidence_from_vote(votes, chosen) -> list[float]
```

Per-token agreement fraction across vote streams.

```python
from hypofuse.confidence import temperature_scale
out = temperature_scale([1.0, 2.0, 3.0], temperature=1.0)
assert abs(sum(out) - 1.0) < 1e-9
```

---

### Additional calibration features

#### `brier_score(probs, labels) -> float`

Mean squared error between predicted probabilities and binary labels.
Lower is better; perfect predictions yield 0.0.

#### `log_loss(probs, labels, eps=1e-15) -> float`

Cross-entropy with clipping to avoid `log(0)`.

#### `reliability_curve(probs, labels, bins=10)`

Returns a `ReliabilityCurve` dataclass with `bin_edges`, `mean_predicted`,
`observed_frequency`, and `counts` tuples.

#### `LogisticCalibrator` / `fit_logistic(...)`

Platt-style logistic calibrator fitted via full-batch gradient descent.

#### `fit_temperature(scores, labels) -> float`

Deterministic grid search for the temperature minimising NLL.

#### `utterance_confidence(..., aggregation)`

Aggregation modes: `mean`, `min`, `geometric`,
`length_normalized`.

#### `calibration_report(probs, labels) -> dict`

Keys: `binning`, `bins`, `brier_score`, `ece`,
`log_loss`.

#### `synthetic_calibration_set(n, seed, sharpness)`

Returns `(scores, labels, p_true)`. Synthetic only.

## hypofuse.convert

### Per-schema converters

| Function | Schema | Output |
|---|---|---|
| `nbest_from_row` | hypofuse.nbest | NBestList |
| `reference_from_row` | hypofuse.reference | ReferenceTranscript |
| `system_from_row` | hypofuse.system | SystemMetadata |
| `fusion_run_from_row` | hypofuse.fusion_run | FusionRun |
| `report_from_row` | hypofuse.report | ReportRecord |

Each has a matching `*_to_row` function.

### Dispatch and migration

`convert_row(row)` dispatches by `row["schema"]`.
`migrate_row(row, to_version=N)` handles nbest v1->v2.
`migrate_manifest(rows, to_version=N)` migrates all rows.

---

**Note:** There is no `hypofuse.neural` module in the current release.
A future version may provide a neural rescoring interface requiring the
`torch` optional dependency (`pip install hypofuse[torch]`). Any such
module would operate on synthetic demonstration data only.

---

## hypofuse.analysis

### Data classes

```python
@dataclass(frozen=True)
class UtteranceScore:
    utterance_id: str
    reference: tuple[str, ...]
    hypothesis: tuple[str, ...]
    speaker_group: str = ""
    noise_db: float = 0.0
    duration_s: float = 0.0
    intent_domain: str = ""
```

Methods: `error_breakdown() -> (subs, dels, ins)`, property `is_perfect`.

```python
@dataclass(frozen=True)
class SliceMetric:
    key: str
    count: int
    cer: float
    wer: float

@dataclass(frozen=True)
class ComparisonRow:
    system_a: str
    system_b: str
    delta_cer: float
    delta_wer: float
    cer_ci_low: float
    cer_ci_high: float
    wer_ci_low: float
    wer_ci_high: float
    n_pairs: int

@dataclass(frozen=True)
class GroupBiasRow:
    group: str
    count: int
    wer: float
    wer_ci_low: float
    wer_ci_high: float
```

### Slicing functions

```python
slice_by_duration(items, boundaries=(1.0, 3.0, 10.0)) -> dict[str, list[UtteranceScore]]
slice_by_field(items, field_name) -> dict[str, list[UtteranceScore]]
slice_by_quantiles(scores, field, *, buckets=4) -> dict[str, list[UtteranceScore]]
slice_metrics(slices, use_cer=False) -> list[SliceMetric]
```

`slice_by_field` accepts `"speaker_group"`, `"intent_domain"`, `"noise"`.
Raises `ValueError` for unknown field names.

### Substitution mining

```python
substitution_pairs(items, *, top_n=None, directional=True,
                   min_count=1, level="word") -> Counter[tuple[str, str]]
```

`level="char"` joins tokens before alignment for CJK analysis.

### Bootstrap and comparison

```python
paired_bootstrap_ci(system_a, system_b, *, n_bootstrap=1000,
                    confidence=0.95, seed=0) -> ComparisonRow
paired_bootstrap_p_value(a_scores, b_scores, *, seed=0,
                         iterations=1000) -> float
group_bias_table(scores, field, *, n_bootstrap=1000,
                 confidence=0.95, seed=0) -> list[GroupBiasRow]
```

p-value uses add-one smoothing: `p = (count_le + 1) / (iterations + 1)`.
Raises `ValueError` for `iterations < 1` or mismatched lengths.

### Corpus aggregation

```python
corpus_error_rate(scores, average="micro") -> float
```

`"micro"` = total errors / total reference tokens.
`"macro"` = mean of per-utterance rates.

### Report rendering

```python
report_to_markdown(slices, comparisons, *, meta=None) -> str
report_to_jsonl(slices, comparisons) -> str
```

Markdown escapes pipe, backslash, and newline in slice keys.

---

## hypofuse.fixtures

### `FixtureConfig` (frozen dataclass)

```python
@dataclass(frozen=True)
class FixtureConfig:
    n_utterances: int = 20
    n_best: int = 3
    substitution_rate: float = 0.05
    insertion_rate: float = 0.02
    deletion_rate: float = 0.02
    speaker_groups: tuple[str, ...] = ("A", "B", "C")
    intents: tuple[str, ...] = ("greeting", "qa", "command")
    noise_db_range: tuple[float, float] = (-10.0, 35.0)
    duration_range: tuple[float, float] = (0.5, 12.0)
    seed: int = 0
    confusables: tuple[tuple[str, str], ...] = ()
    group_bias: Mapping[str, float] = field(default_factory=dict)
    noise_sensitivity: float = 0.0
    rank_decay: float = 0.0
```

`validate()` raises `ValueError` for rate sums exceeding 1.0,
unknown group bias keys, negative rank decay, or out-of-range
noise sensitivity.

### `FixtureUtterance` (frozen dataclass)

```python
@dataclass(frozen=True)
class FixtureUtterance:
    utterance_id: str
    reference: tuple[str, ...]
    hypotheses: tuple[tuple[str, ...], ...]
    speaker_group: str
    intent_domain: str
    noise_db: float
    duration_s: float
    acoustic_log10s: tuple[float, ...] = ()
    lm_log10s: tuple[float, ...] = ()
```

### `generate_fixture(cfg: FixtureConfig) -> list[FixtureUtterance]`

Deterministic generation via seeded RNG. All output is synthetic.

### `as_manifest_dicts(fixtures, config=None) -> list[dict[str, Any]]`

Renders fixtures as manifest dicts (nbest + reference + system rows).

---

## hypofuse.timings

### `TokenTiming` (frozen dataclass)

```python
@dataclass(frozen=True)
class TokenTiming:
    token: str
    start_s: float
    end_s: float
```

`validate()` rejects NaN, negative values, and `end_s < start_s`.
Property `duration_s`.

### `TimingTrack` (frozen dataclass)

```python
@dataclass(frozen=True)
class TimingTrack:
    tokens: tuple[TokenTiming, ...] = ()
    tolerance_s: float = 1e-9
```

Properties: `total_s`, `speaking_ratio()`. `validate()` rejects
overlapping tokens and non-finite values.

### Functions

```python
align_timings(reference_track, hypothesis_track) -> tuple[TimingAlignmentStep, ...]
interpolate_gaps(track, max_gap_s) -> TimingTrack
snap_to_grid(track, frame_s=0.01) -> TimingTrack
duration_buckets(total_s, edges=(1.0, 3.0, 8.0)) -> str
timing_report(tracks) -> TimingReport
from_manifest_row(row) -> TimingTrack
to_manifest_row(track) -> dict[str, Any]
```

---

## hypofuse.config

### Generic serialization

```python
to_dict(config) -> dict[str, Any]
from_dict(cls, data) -> T
load_config(cls, path) -> T
dump_config(config, path) -> None
config_hash(config) -> str
merge_configs(base, overrides) -> T
describe(config) -> str
```

`to_dict` / `from_dict` handle nested frozen dataclasses, tuples, and
lists. `from_dict` raises `ValueError` for unknown or missing keys.
`merge_configs` supports dotted paths (e.g. `"inner.x"`).

### `HypofuseRunConfig` (frozen dataclass)

```python
@dataclass(frozen=True)
class HypofuseRunConfig:
    normalization: dict[str, Any]
    fusion: dict[str, Any]
    fixture: dict[str, Any]
    lm_order: int = 3
    lm_weight: float = 0.5
    seed: int = 0
```

`validate()` checks `lm_order >= 1`, finite `lm_weight`, `seed >= 0`.
`HypofuseRunConfig.from_parts(...)` builds from config instances.

---

## hypofuse.runmeta

### `RunMetadata` (frozen dataclass)

```python
@dataclass(frozen=True)
class RunMetadata:
    hypofuse_version: str
    python_version: str
    platform: str
    numpy_version: str
    seed: int
    config_hash: str
    created_from: str = ""
    torch_version: str | None = None
    stamped_at: str | None = None
```

### Functions

```python
capture(seed, config=None, label="") -> RunMetadata
stamp(meta, when: datetime) -> RunMetadata
to_dict(meta) -> dict
from_dict(data) -> RunMetadata
to_json(meta) -> str
from_json(text) -> RunMetadata
embed(report_rows, meta) -> list[dict]
verify(rows, meta) -> bool
fingerprint_files(paths) -> dict[str, str]
```

`stamp` does not read wall-clock time; the caller supplies the datetime.
`embed` adds a `"run"` key to each row without mutating the originals.

---

## hypofuse.util

```python
stable_hash(obj) -> str        # SHA-256 of canonical JSON
seeded(seed) -> random.Random  # deterministic RNG (version=2)
relative_within(base, target) -> str  # raises ValueError on escape
safe_join(base, *parts) -> Path       # raises ValueError on escape
json_default(value) -> Any            # numpy/dataclass/Path handler
write_jsonl(path, rows) -> None
read_jsonl(path) -> list[dict]        # raises ValueError on parse error
env_flag(name, default=False) -> bool
```

---

## hypofuse.exceptions

| Exception | Parent | Meaning |
|---|---|---|
| `HypofuseError` | `Exception` | Base class |
| `SchemaError` | `HypofuseError` | Manifest validation failure |
| `DuplicateIdError` | `HypofuseError` | Duplicate identifier |
| `AlignmentError` | `HypofuseError` | Malformed alignment input |
| `FusionError` | `HypofuseError` | Unsolvable fusion config |
| `LanguageModelError` | `HypofuseError` | N-gram training/load failure |
| `CalibrationError` | `HypofuseError` | Inconsistent calibration input |
