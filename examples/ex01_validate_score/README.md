# ex01: validate and score

Generate a small synthetic fixture with `FixtureConfig`, write JSONL
manifests, validate them (including a deliberately invalid row to show
the rejection path), normalize text, and compute CER / WER.

## Usage

```
python examples/ex01_validate_score/run.py <outdir>
```

## Artifacts

- `nbest.jsonl` -- n-best hypotheses (3 utterances, 2-best)
- `reference.jsonl` -- reference transcripts
