# hypofuse

语音识别后处理工具包（Python 3.11+），提供 n-best 融合、对齐、语言模型重打分、置信度校准和误差分析。

## 项目简介

hypofuse 实现了从假设列表到融合结果的完整后处理流水线。核心思路来自 ROVER（Fiscus 1997）：对多个识别系统的输出做多假设对齐，再通过投票或加权策略选出融合结果。在此基础上增加了 n-gram 语言模型重打分、confusion network 构建、置信度校准和分片误差分析。

## 功能特性

- **Manifest schemas** -- JSONL 格式的 n-best 列表、参考文本、系统元数据和融合结果，带 schema 版本字段和严格校验
- **文本归一化** -- Unicode NFC、全半角统一、标点去除、大小写折叠、数字归一化、中英文分词
- **逆文本归一化（ITN）** -- 英文和中文数字短语与阿拉伯数字互转，支持小数、负数、百分数和单位
- **编辑对齐** -- Levenshtein 动态规划对齐，支持自定义代价矩阵和 Sakoe-Chiba 带状约束
- **多假设渐进对齐** -- 将 N 个假设对齐到统一的 token grid，支持 left-pivot 和排序模式
- **ROVER 融合** -- majority、score_weighted、lm_weighted 三种投票策略，可配置 tie-breaking
- **Confusion network** -- 从对齐 grid 构建 confusion network，每个 slot 的 posterior 归一化到 1
- **N-gram 语言模型** -- Katz backoff + Jelinek-Mercer 平滑，ARPA 格式导入导出
- **N-best 重打分** -- shallow fusion 风格，组合声学得分和 LM 得分，支持权重扫描
- **置信度校准** -- temperature scaling 和分段线性校准，ECE 计算
- **误差分析** -- 按 speaker group、intent domain、noise level、duration 分片，paired bootstrap 置信区间
- **合成 fixture** -- 确定性合成 ASR-like 数据，带已知 ground truth，用于测试和演示
- **CLI** -- 10 个子命令，覆盖 validate、normalize、score、align、fuse、rescore、calibrate、analyze、report、demo
- **Token timing** -- per-token 时间戳处理、对齐、插值、grid 量化和统计报告

## 数据声明

- 所有捆绑数据均为合成数据（synthetic），不附带真实语音或转录
- 不附带、不下载任何预训练权重
- 可选的 `torch` extra 仅用于演示置信度模型，且仅使用 CPU
- 项目中没有任何地方声称真实基准测试结果

## 安装

```bash
pip install hypofuse
```

开发依赖：

```bash
pip install "hypofuse[dev]"
```

可选 torch extra（仅 CPU，用于演示置信度模型）：

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu "hypofuse[torch]"
```

## 快速开始

运行合成数据演示：

```bash
hypofuse demo --out demo_out --utterances 5 --n-best 3 --seed 0
```

使用库 API：

```python
from hypofuse.normalize import normalize, NormalizationConfig
from hypofuse.alignment import word_error_rate
from hypofuse.multi_align import progressive_align
from hypofuse.fusion import fuse, FusionConfig

ref = normalize("The cat sat on the mat")
hyp = normalize("the cat sit on a mat")
wer = word_error_rate(ref.split(), hyp.split())

grid = progressive_align([
    ["the", "cat", "sat"],
    ["the", "cat", "sit"],
    ["a", "cat", "sat"],
])
result = fuse(grid, config=FusionConfig(policy="majority"))
```

## CLI 子命令

| 子命令 | 说明 | 示例 |
|---|---|---|
| `validate` | 校验 JSONL manifest | `hypofuse validate data.jsonl --schema hypofuse.nbest` |
| `normalize` | 归一化文本对 | `hypofuse normalize --reference "Hello!" --hypothesis "hello" --language en` |
| `score` | 计算 CER / WER | `hypofuse score --nbest n.jsonl --reference r.jsonl --metric both` |
| `align` | 多假设对齐 | `hypofuse align --nbest n.jsonl --json` |
| `fuse` | 融合 n-best | `hypofuse fuse --nbest n.jsonl --policy majority --tie-break lexicographic` |
| `rescore` | LM 重打分 | `hypofuse rescore --nbest n.jsonl --arpa lm.arpa --lm-weight 0.5 --top 1` |
| `calibrate` | 置信度校准 | `hypofuse calibrate --scores "0.1,0.5,0.9" --temperature 1.5` |
| `analyze` | 分片误差分析 | `hypofuse analyze --nbest n.jsonl --reference r.jsonl --slice-by speaker_group` |
| `report` | 生成报告 | `hypofuse report --nbest n.jsonl --reference r.jsonl --format markdown --out report.md` |
| `demo` | 端到端演示 | `hypofuse demo --out demo_out --utterances 5 --seed 0` |

退出码：`0` 成功，`2` 用户错误（文件缺失、manifest 无效、参数错误）。

## 库 API 概览

### 归一化与评分

```python
from hypofuse.normalize import normalize, normalize_pair, NormalizationConfig

cfg = NormalizationConfig(
    case_fold=True, strip_punctuation=True, language_hint="en",
)
ref, hyp = normalize_pair("Hello, World!", "hello world", cfg)
```

### 对齐与融合

```python
from hypofuse.multi_align import progressive_align
from hypofuse.fusion import fuse, FusionConfig

grid = progressive_align([
    ["a", "b", "c"],
    ["a", "b", "d"],
    ["a", "x", "c"],
])
result = fuse(grid, config=FusionConfig(policy="majority", tie_break="lexicographic"))
# result.tokens -- 融合后的 token 序列
# result.confidences -- 每个 token 的一致性分数
```

### Confusion Network

```python
from hypofuse.confusion import build_confusion_network

network = build_confusion_network(grid)
one_best = network.one_best()
network.validate()
```

### N-gram LM 与重打分

```python
from hypofuse.ngram import NgramLM
from hypofuse.rescore import ScoredHypothesis, rescore_nbest

lm = NgramLM.train([["the", "cat", "sat"], ["the", "dog", "ran"]], order=3)
arpa_text = lm.to_arpa()

hyps = [
    ScoredHypothesis.from_tokens(["the", "cat", "sat"], acoustic_log10=-5.0),
    ScoredHypothesis.from_tokens(["the", "dog", "ran"], acoustic_log10=-6.0),
]
ranked = rescore_nbest(hyps, lm, lm_weight=0.5, acoustic_weight=1.0)
```

### 置信度校准

```python
from hypofuse.confidence import temperature_scale, expected_calibration_error

calibrated = temperature_scale([0.1, 0.5, 0.9], temperature=1.5)
```

### 误差分析报告

```python
from hypofuse.analysis import (
    UtteranceScore, slice_by_field, slice_metrics, report_to_markdown,
)

items = [
    UtteranceScore(
        utterance_id="u0",
        reference=("a", "b"),
        hypothesis=("a", "c"),
        speaker_group="A",
    ),
]
slices = slice_by_field(items, "speaker_group")
metrics = slice_metrics(slices)
md = report_to_markdown(metrics, [])
```

### 合成 Fixture

```python
from hypofuse.fixtures import FixtureConfig, generate_fixture, as_manifest_dicts

cfg = FixtureConfig(n_utterances=10, n_best=3, seed=42)
utterances = generate_fixture(cfg)
rows = as_manifest_dicts(utterances)
```
