# Architecture

## Module Map

```
manifests/            JSONL I/O, schema validation, path safety
  __init__.py         schema constants, KNOWN_SCHEMAS, SCHEMA_VERSION
  jsonl.py            read_manifest / write_manifest
  validate.py         validate_record / reject_duplicate_ids
  paths.py            normalize_audio_path / safe_audio_join
  nbest.py            NBestList / NBestHypothesis
  reference.py        ReferenceTranscript
  system.py           SystemMetadata
  fusion_run.py       FusionRun / FusionArc
  reporter.py         ReportRecord
normalize.py          normalize / tokenize / normalize_pair / NormalizationConfig
itn.py                words_to_digits / digits_to_words / ItnConfig
alignment.py          edit_alignment / word_error_rate / character_error_rate
multi_align.py        progressive_align / TokenGrid / grid_row / grid_is_consistent
fusion.py             fuse / FusionConfig / FusionResult / fusion_invariants
confusion.py          build_confusion_network / ConfusionNetwork / confusion_to_json
ngram.py              NgramLM.train / NgramLM.from_arpa / to_arpa / perplexity
rescore.py            rescore_nbest / shallow_fusion_score / lm_weight_sweep
confidence.py         temperature_scale / piecewise_calibrate / expected_calibration_error
analysis.py           slice_by_field / slice_metrics / report_to_markdown / paired_bootstrap_ci
fixtures.py           FixtureConfig / generate_fixture / as_manifest_dicts
timings.py            TokenTiming / TimingTrack / align_timings / snap_to_grid
config.py             to_dict / from_dict / load_config / dump_config / HypofuseRunConfig
runmeta.py            RunMetadata / capture / stamp / embed / verify
util.py               stable_hash / seeded / write_jsonl / read_jsonl
exceptions.py         HypofuseError and typed subclasses
cli.py                main / 10 subcommands (validate..demo)
```

## Data Flow

A typical fusion run follows this path through the modules:

```
JSONL manifest (nbest + reference)
  |
  v
manifests/jsonl.read_manifest   -- parse + validate schema + reject duplicates
  |
  v
normalize.normalize             -- NFC, fullwidth, punct, case, whitespace
  |
  v
multi_align.progressive_align   -- pairwise edit alignment -> TokenGrid
  |
  v
fusion.fuse                     -- per-column voting -> FusionResult
  |
  v
confusion.build_confusion_network  -- optional: posterior per slot
  |
  v
rescore.rescore_nbest           -- optional: LM re-ranking
  |
  v
confidence.temperature_scale    -- optional: calibrate scores
  |
  v
analysis.slice_metrics          -- per-slice CER / WER
  |
  v
analysis.report_to_markdown    -- render report
```

## Invariants

1. **Fusion never invents tokens.** Every token in the fused output appears in at least one input hypothesis. Enforced by `fusion_invariants()` and tested in `test_fusion_invariants.py`.

2. **Alignment paths are monotone and complete.** The edit alignment consumes every token of both sequences exactly once; back-pointers are non-decreasing. Tested in `test_alignment_basic.py` and `test_multi_align_consistent.py`.

3. **LM export/import is lossless for count data.** `NgramLM.to_arpa()` followed by `NgramLM.from_arpa()` preserves n-gram counts. Tested in `test_ngram_arpa.py`.

4. **Calibration preserves rank order.** Temperature scaling with T > 0 preserves the relative ordering of scores. Tested in `test_confidence_ranking.py`.

5. **Reports escape special characters.** Markdown report cells escape pipe, backslash and newline. Tested in `test_analysis_escaping.py`.

6. **Confusion network posteriors sum to 1.** Each slot's arc posteriors normalize to 1.0 within tolerance. Tested in `test_confusion_validate.py`.

7. **Manifest round-trips are stable.** Writing and re-reading a manifest produces identical records. Tested in `test_manifests_roundtrip.py`.

8. **Grid consistency.** Back-pointers reference valid columns and `grid_row` reconstructs original tokens. Tested in `test_multi_align_consistent.py`.
