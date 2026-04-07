# CLI Command Reference

Install: `pip install hypofuse` (or `pip install -e ".[dev]"` for development).
Entry point: `hypofuse`. Version: `0.1.0`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | User error (missing file, invalid manifest, bad flag, `HypofuseError`, `ValueError`) |
| 2 | argparse error for unrecognized/missing flags (standard argparse behavior) |

## Global flags

- `--version` -- print version and exit 0.
- `--help` / `-h` -- print usage.

## Subcommands

### `validate`

```
hypofuse validate PATH [--schema NAME] [--json]
```

| Flag | Default | Description |
|---|---|---|
| `path` | (required) | Path to JSONL manifest |
| `--schema` | `None` | Expected schema name |
| `--json` | off | Emit JSON `{"valid": true, "count": N}` |

```
$ hypofuse validate nbest.jsonl
validated 10 records
```

### `normalize`

```
hypofuse normalize --reference TEXT --hypothesis TEXT [--language en|zh] [--keep-case] [--keep-punct] [--json]
```

| Flag | Default | Description |
|---|---|---|
| `--reference` | (required) | Reference text |
| `--hypothesis` | (required) | Hypothesis text |
| `--language` | `"en"` | Language hint (`"en"` or `"zh"`) |
| `--keep-case` | off | Disable case folding |
| `--keep-punct` | off | Keep punctuation |
| `--json` | off | Emit JSON (default output is already JSON) |

Output: `{"reference": "...", "hypothesis": "..."}`

### `score`

```
hypofuse score --nbest PATH --reference PATH [--metric cer|wer|both] [--no-normalize] [--json]
```

| Flag | Default | Description |
|---|---|---|
| `--nbest` | (required) | N-best manifest path |
| `--reference` | (required) | Reference manifest path |
| `--metric` | `"both"` | Metric to compute |
| `--no-normalize` | off | Skip text normalization |
| `--json` | off | Emit JSON output |

Output: `{"n": 5, "cer": 0.12, "wer": 0.15}`

### `align`

```
hypofuse align --nbest PATH [--json]
```

Output: `{"utterance_id": "...", "width": 4, "depth": 3}`

### `fuse`

```
hypofuse fuse --nbest PATH [--policy majority|score_weighted|lm_weighted] [--tie-break lexicographic|first] [--json]
```

| Flag | Default | Description |
|---|---|---|
| `--policy` | `"majority"` | Voting policy |
| `--tie-break` | `"lexicographic"` | Tie-breaking strategy |

Output: `{"utterance_id": "...", "tokens": [...], "confidences": [...]}`

### `rescore`

```
hypofuse rescore --nbest PATH (--arpa PATH | --corpus PATH) [--order N] [--lm-weight W] [--acoustic-weight W] [--top N] [--json]
```

| Flag | Default | Description |
|---|---|---|
| `--arpa` | -- | ARPA LM file (mutually exclusive with `--corpus`) |
| `--corpus` | -- | Text corpus (one sentence per line) |
| `--order` | `3` | N-gram order for corpus training |
| `--lm-weight` | `0.5` | LM weight for shallow fusion |
| `--acoustic-weight` | `1.0` | Acoustic weight |
| `--top` | `1` | Number of top hypotheses to show |

### `calibrate`

```
hypofuse calibrate (--scores CSV | --scores-file PATH) [--temperature T] [--method temperature|piecewise] [--json]
```

| Flag | Default | Description |
|---|---|---|
| `--scores` | -- | Comma-separated score list |
| `--scores-file` | -- | File with one score per line |
| `--temperature` | `1.0` | Temperature parameter |
| `--method` | `"temperature"` | Calibration method |

### `analyze`

```
hypofuse analyze --nbest PATH --reference PATH [--slice-by FIELD] [--buckets N] [--json]
```

| Flag | Default | Description |
|---|---|---|
| `--slice-by` | `"speaker_group"` | Field: `speaker_group`, `intent_domain`, `noise_db`, `duration_s` |
| `--buckets` | `3` | Duration bucket count |

### `report`

```
hypofuse report --nbest PATH --reference PATH [--format markdown|jsonl] [--out PATH] [--compare ID]
```

| Flag | Default | Description |
|---|---|---|
| `--format` | `"markdown"` | Output format |
| `--out` | stdout | Output file path |
| `--compare` | `None` | System ID for comparison |

### `demo`

```
hypofuse demo [--out DIR] [--utterances N] [--n-best N] [--seed N]
```

| Flag | Default | Description |
|---|---|---|
| `--out` | `"."` | Output directory |
| `--utterances` | `5` | Synthetic utterance count |
| `--n-best` | `3` | N-best list size |
| `--seed` | `0` | Random seed |

Fully offline. Generates synthetic manifests and a summary report.

```
$ hypofuse demo --utterances 5 --n-best 2 --seed 42 --out /tmp/demo
demo: wrote manifests and report to /tmp/demo
  CER=<value> WER=<value> fused=5
```

Run the command in-repo to see the actual CER/WER values.
