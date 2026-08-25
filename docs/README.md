# hypofuse documentation index

| Document | Contents |
|---|---|
| [api.md](api.md) | Reference for every public symbol: manifests, normalization, ITN, alignment, fusion, confusion networks, language models, rescoring, confidence, analysis, fixtures, timings, config, run metadata and conversion. |
| [cli.md](cli.md) | Every `hypofuse` subcommand with its flags, defaults, exit codes and example invocations. |
| [usage.md](usage.md) | End-to-end workflows: validate + normalize + score, ROVER fusion + confusion network, rescoring + calibration + report, plus configuration and extension notes. |
| [schemas.md](schemas.md) | The versioned JSONL schemas, their fields, the migration policy and the rejection rules. |
| [metrics.md](metrics.md) | CER/WER definitions as implemented, the edit-distance cost model, micro vs macro averaging and the empty-reference policy. |
| [fusion.md](fusion.md) | ROVER voting policies, tie-breaks, null handling, fused confidence and the confusion-network relation. |
| [lm.md](lm.md) | N-gram counting, Katz backoff, Jelinek-Mercer interpolation, lambda fitting, perplexity, OOV policy and the ARPA export/import guarantees. |
| [calibration.md](calibration.md) | Confidence sources, temperature scaling, the logistic and piecewise calibrators, ECE/reliability bins and the rank-preservation guarantees. |
| [error-analysis.md](error-analysis.md) | Slicing rules, substitution mining, paired bootstrap confidence intervals and p-values, report formats and escaping. |
| [fixtures.md](fixtures.md) | The synthetic fixture factory: every config field, the error model and how known ground truth is asserted in tests. |
| [neural.md](neural.md) | The optional CPU-only torch confidence model, its features and the synthetic dataset it is demonstrated on. |
| [reproducibility.md](reproducibility.md) | Seeds, the pinned lockfile, versioned artifacts, run metadata and the release process. |
| [sources.md](sources.md) | Upstream references used for ideas and format facts, and what was taken from each. |

Top-level files: [README](../README.md) (Chinese overview), [ARCHITECTURE](../ARCHITECTURE.md),
[CHANGELOG](../CHANGELOG.md), [CONTRIBUTING](../CONTRIBUTING.md), [SECURITY](../SECURITY.md),
[LICENSE](../LICENSE) (MIT).

Runnable offline examples live in [`examples/`](../examples/README.md); `make examples` runs all
three, and `make help` lists every developer target.

All data used by the examples, tests and documentation is synthetic and generated in-process.
Nothing here downloads a model, a dataset or any other artifact, and no real ASR benchmark
result is claimed.
