# Error Analysis: Slicing, Bootstrap, and Reports

## Slice fields and bucketing

### Duration slicing

`slice_by_duration(items, boundaries=(1.0, 3.0, 10.0))` partitions
utterances into buckets based on `duration_s`:

| Bucket label | Range |
|---|---|
| `<1.0s` | `duration_s < 1.0` |
| `<3.0s` | `1.0 <= duration_s < 3.0` |
| `<10s` | `3.0 <= duration_s < 10.0` |
| `>10s` | `duration_s >= 10.0` |

An `"all"` bucket always contains every item. Custom boundaries
produce labels like `"<{b}s"` for the lowest bucket and
`"{prev}-{b}s"` for intermediate ones.

### Field slicing

`slice_by_field(items, field_name)` accepts:
- `"speaker_group"` -- groups by `speaker_group` field (empty -> `"unknown"`).
- `"intent_domain"` -- groups by `intent_domain` field.
- `"noise"` -- categorizes by `noise_db`: `< 0` -> `"quiet"`,
  `< 20` -> `"moderate"`, else `"noisy"`.

Raises `ValueError` for unknown field names.

### Quantile slicing

`slice_by_quantiles(scores, field, buckets=4)` partitions by
nearest-rank quantile edges on any numeric field. Labels are
`"q0"`, `"q1"`, etc. If all values are equal, returns a single
`"all"` bucket.

## Substitution mining

`substitution_pairs(items, *, top_n=None, directional=True, min_count=1, level="word")`

- `directional=True`: counts `(ref_token, hyp_token)` in order.
- `directional=False`: collapses `(a, b)` and `(b, a)` into a sorted pair.
- `min_count`: drops pairs below threshold.
- `top_n`: keeps only the N most frequent (ties broken by count
  descending, then lexicographic on the key tuple).
- `level="char"`: joins tokens into characters before alignment.

## Paired bootstrap CI

`paired_bootstrap_ci(system_a, system_b, *, n_bootstrap=1000, confidence=0.95, seed=0)`

1. Resample utterance indices with replacement, `n_bootstrap` times.
2. Compute mean CER and WER for both systems on each resample.
3. Record `delta = mean_B - mean_A` for each iteration.
4. CI bounds are percentile quantiles: `alpha = (1 - confidence) / 2`.

Returns a `ComparisonRow` with `delta_cer`, `delta_wer`, and CI bounds.
Deterministic via the `seed` parameter.

## Paired bootstrap p-value

`paired_bootstrap_p_value(a_scores, b_scores, *, seed=0, iterations=1000)`

One-sided p-value that system A is not better than B:
1. Compute per-utterance WER for both systems.
2. Resample indices, compute `mean_diff = mean(wer_b - wer_a)`.
3. Count how often `mean_diff <= 0` (A's advantage does not hold).
4. Apply add-one smoothing: `p = (count_le + 1) / (iterations + 1)`.

A small p-value indicates A is reliably better. Raises `ValueError`
for `iterations < 1` or mismatched list lengths.

## Report formats

### Markdown

`report_to_markdown(slices, comparisons, *, meta=None)` produces:
- `# Error Analysis Report`
- Optional `## Run` section with sorted key-value metadata.
- `## Slices` table with columns: Slice, Count, CER, WER.
- `## System comparisons` table with CI brackets.

Escaping: pipe (`|`), backslash (`\`), and newline are escaped in
slice keys. HTML entities in keys are also escaped.

### JSONL

`report_to_jsonl(slices, comparisons)` emits one JSON record per line:
- Slice rows: `{"kind": "slice", "schema": "hypofuse.report", ...}`
- Comparison rows: `{"kind": "comparison", "schema": "hypofuse.report", ...}`

## Run metadata embedding

When `meta` is provided to `report_to_markdown`, a `## Run` section
lists metadata keys in sorted order. The `runmeta.embed()` function
adds a `"run"` key to JSONL report rows for provenance tracking.
