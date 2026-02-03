# ex03: rescore and calibrate

Train a tiny n-gram LM on an in-script synthetic corpus, rescore an
n-best list with a small LM-weight sweep, calibrate per-token
confidences with `temperature_scale`, compute ECE / reliability bins,
and write a Markdown error-analysis report.

> The numbers come from synthetic data with a constructed relationship,
> not from a real ASR system or benchmark.

## Usage

```
python examples/ex03_rescore_calibrate/run.py <outdir>
```

## Artifacts

- `report.md` -- Markdown error-analysis report
