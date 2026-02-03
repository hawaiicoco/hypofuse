# Examples

Offline, deterministic examples that exercise the hypofuse pipeline on
synthetic data generated in-process.

> **All data is synthetic.** No downloads, no network access, no real ASR
> output. Every number comes from a seeded generator with a known ground
> truth.

## Catalogue

| Example | What it shows |
|---|---|
| [ex01\_validate\_score](ex01_validate_score/) | Fixture generation, manifest validation (including a deliberately invalid row), normalization, CER/WER scoring |
| [ex02\_rover\_confusion](ex02_rover_confusion/) | Progressive alignment, ROVER fusion with two policies, confusion network construction, 1-best extraction |
| [ex03\_rescore\_calibrate](ex03_rescore_calibrate/) | N-gram LM training, LM-weight sweep, temperature scaling, ECE / reliability bins, Markdown report |

## Running

Run a single example:

```
python examples/ex01_validate_score/run.py <outdir>
```

Run all examples at once:

```
python scripts/run_examples.py [<basedir>]
```

Each example exits non-zero on failure.
