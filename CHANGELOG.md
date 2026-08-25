# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-09

### Fixed

- Include the hatchling build backend in the dev extra so wheel builds work in freshly provisioned environments.
- Stop pinning the mypy target version; typecheck runs against current numpy stubs on Python 3.13.

## [0.1.1] - 2026-09

### Added

- Public `__all__` surface re-exporting primary names from every module.
- `hypofuse.neural` optional torch confidence demonstration module.
- `hypofuse.itn` inverse text normalization.
- `hypofuse.timings` token timing handling.
- `hypofuse.config` declarative configuration.
- `hypofuse.runmeta` reproducible run metadata.
- `hypofuse.convert` strict manifest conversion.
- Manifest schema versions; nbest v1 to v2 migration.
- ARPA probability tables with `prob(method="auto")`.
- N-gram pruning, coverage, interpolation, EM lambda fitting.
- Extra fusion policies, null policies, tie-break rules.
- Confidence additions: brier, log loss, logistic calibrator, calibration report.
- Analysis additions: micro/macro corpus rates, quantile slices, group bias table.
- Normalization presets and options.
- Alignment cost models and banded alignment.
- Insertion penalty and LM-weight monotonicity check.
- `make examples` and `make check-package` targets.
- CI workflow runs examples after build.

### Changed

- Token grid columns are rectangular.
- `digits_to="spoken"` emits number words.
- Duplicate-id messages name both line numbers.
- JSONL writer is atomic and compact.

### Fixed

- ARPA backoff weights no longer collapse to 1e-12.
- Bootstrap resampling reuses per-utterance rates.
- Normalization is idempotent for inputs with stripped combining marks.

## [0.1.0] - 2026-09

### Added

- **Manifests package**: JSONL read/write with schema validation, duplicate-id
  rejection, and audio-path safety checks. Schemas: `hypofuse.nbest`,
  `hypofuse.reference`, `hypofuse.system`, `hypofuse.fusion_run`,
  `hypofuse.report`.
- **Text normalization**: NFC, fullwidth unification, punctuation stripping,
  case folding, digit handling, auto tokenization (word/char).
- **Inverse text normalization (ITN)**: English and Chinese number phrases,
  decimals, negatives, percent, and unit expressions.
- **Edit alignment**: Levenshtein DP with custom costs, Sakoe-Chiba banding,
  WER/CER/symmetric rates, corpus micro/macro averaging, error breakdown.
- **Multi-hypothesis alignment**: Progressive alignment into a rectangular
  token grid with gap handling and consistency checks.
- **ROVER fusion**: Majority, score-weighted, and LM-weighted voting policies
  with configurable tie-breaking. Fused confidence as agreement fraction.
- **Confusion networks**: Vote-mass posterior estimation, pivot selection,
  1-best extraction, versioned JSON serialization, ROVER consistency checks.
- **N-gram language model**: Training from tokenized sentences, Katz backoff,
  Jelinek-Mercer interpolation, perplexity, ARPA export/import.
- **N-best rescoring**: LM rescoring, shallow fusion, weight sweep.
- **Confidence and calibration**: Temperature scaling, piecewise calibrator,
  ECE, reliability bins, token/utterance confidence.
- **Error analysis**: Duration/field/quantile slicing, substitution mining,
  paired bootstrap CI and p-value, group bias table, Markdown/JSONL reports.
- **Synthetic fixture factory**: Deterministic data generation with known
  ground truth, configurable error rates, group bias, noise sensitivity.
- **Token timings**: Timing tracks, alignment, gap interpolation,
  grid snapping, duration buckets, summary statistics.
- **Configuration**: Generic dataclass serialization, JSON I/O, content
  hashing, dotted-path merging, `HypofuseRunConfig`.
- **Run metadata**: Version capture, deterministic stamping, embedding
  into report rows, file fingerprinting.
- **CLI**: Ten subcommands (`validate`, `normalize`, `score`, `align`,
  `fuse`, `rescore`, `calibrate`, `analyze`, `report`, `demo`).
  Offline demo generates synthetic data end-to-end.
- **Documentation**: API reference, schema reference, CLI reference,
  metrics definitions, fusion guide, LM guide, calibration guide,
  error analysis guide, fixtures guide, reproducibility guide,
  upstream references.
- **Project files**: Changelog, contributing guide, security policy.

### Changed

- Nothing (initial release).

### Fixed

- Nothing (initial release).
