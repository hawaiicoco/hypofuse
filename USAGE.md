# Usage

## Workflows

### 1. Validate, Normalize, Score

Validate manifest files, then compute error rates:

```bash
# Validate manifests
hypofuse validate nbest.jsonl --schema hypofuse.nbest
hypofuse validate reference.jsonl --schema hypofuse.reference

# Normalize a text pair
hypofuse normalize --reference "Hello, World!" --hypothesis "hello world" --language en

# Score n-best against references
hypofuse score --nbest nbest.jsonl --reference reference.jsonl --metric both
```

The `score` command outputs JSON with `n` (matched utterance count), `cer`, and `wer` fields.

### 2. ROVER Fusion with Confusion Network

Align and fuse multiple hypotheses:

```bash
# Align hypotheses and print grid dimensions
hypofuse align --nbest nbest.jsonl --json

# Fuse with majority voting
hypofuse fuse --nbest nbest.jsonl --policy majority --tie-break lexicographic
```

Library API for confusion network construction:

```python
from hypofuse.multi_align import progressive_align
from hypofuse.fusion import fuse, FusionConfig
from hypofuse.confusion import build_confusion_network

hypotheses = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "cat", "sit", "on", "a", "mat"],
    ["a", "cat", "sat", "on", "the", "mat"],
]
grid = progressive_align(hypotheses)
result = fuse(grid, config=FusionConfig(policy="majority"))
network = build_confusion_network(grid)
network.validate()
one_best = network.one_best()
```

### 3. Rescore, Calibrate, Report

Train an LM, rescore hypotheses, calibrate confidence, generate report:

```bash
# Rescore with an ARPA language model
hypofuse rescore --nbest nbest.jsonl --arpa model.arpa --lm-weight 0.5 --top 3

# Rescore by training from a text corpus
hypofuse rescore --nbest nbest.jsonl --corpus corpus.txt --order 3 --lm-weight 0.5

# Calibrate scores with temperature scaling
hypofuse calibrate --scores "0.1,0.3,0.5,0.7,0.9" --temperature 1.5 --json

# Error analysis sliced by speaker group
hypofuse analyze --nbest nbest.jsonl --reference reference.jsonl --slice-by speaker_group

# Generate a markdown report
hypofuse report --nbest nbest.jsonl --reference reference.jsonl --format markdown --out report.md
```

Library API for the same workflow:

```python
from hypofuse.ngram import NgramLM
from hypofuse.rescore import ScoredHypothesis, rescore_nbest
from hypofuse.confidence import temperature_scale

lm = NgramLM.train([["the", "cat", "sat"], ["the", "dog", "ran"]], order=3)
hyps = [
    ScoredHypothesis.from_tokens(["the", "cat", "sat"], acoustic_log10=-5.0),
    ScoredHypothesis.from_tokens(["the", "dog", "ran"], acoustic_log10=-6.0),
]
ranked = rescore_nbest(hyps, lm, lm_weight=0.5, acoustic_weight=1.0)
calibrated = temperature_scale([h.acoustic_log10 for h in ranked], temperature=1.5)
```

## Troubleshooting

### Exit Codes

| 退出码 | 含义 | 常见原因 |
|---|---|---|
| `0` | 成功 | -- |
| `2` | 用户错误 | 文件不存在、manifest schema 不匹配、参数值无效、空输入 |

### 常见错误消息

| 错误消息 | 原因与解决 |
|---|---|
| `file not found: <path>` | 检查文件路径是否正确 |
| `no matched utterance ids` | nbest 和 reference manifest 的 `utterance_id` 没有交集 |
| `no n-best rows found` | manifest 中没有 `schema: "hypofuse.nbest"` 的记录 |
| `expected schema X, got Y` | `--schema` 参数与 manifest 中的 schema 字段不匹配 |
| `empty scores list` | `calibrate` 命令收到了空的分数列表 |
| `duplicate utterance_id` | manifest 文件中存在重复的 `utterance_id` |
