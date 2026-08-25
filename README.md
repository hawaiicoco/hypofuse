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

grid = progressive_align(
    [
        ["the", "cat", "sat"],
        ["the", "cat", "sit"],
        ["a", "cat", "sat"],
    ]
)
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
    case_fold=True,
    strip_punctuation=True,
    language_hint="en",
)
ref, hyp = normalize_pair("Hello, World!", "hello world", cfg)
```

### 对齐与融合

```python
from hypofuse.multi_align import progressive_align
from hypofuse.fusion import fuse, FusionConfig

grid = progressive_align(
    [
        ["a", "b", "c"],
        ["a", "b", "d"],
        ["a", "x", "c"],
    ]
)
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
    UtteranceScore,
    slice_by_field,
    slice_metrics,
    report_to_markdown,
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

## 可复现性与测试

所有随机行为均通过 seed 参数控制。运行时唯一依赖为 `numpy>=1.24`，依赖声明在 `pyproject.toml` 中。

### 构建与测试命令

| 命令 | 说明 |
|---|---|
| `make build` | 构建 wheel |
| `make test` | 运行快速测试（排除 `slow` 和 `model` 标记） |
| `make test-all` | 运行所有测试（含 slow 和 model） |
| `make format-check` | ruff format 检查 + ruff lint |
| `make lint` | ruff check |
| `make typecheck` | mypy 类型检查 |

### 测试标记

- `@pytest.mark.slow` -- 耗时较长的测试，默认 `make test` 跳过
- `@pytest.mark.model` -- 需要可选 `torch` extra 的测试，默认跳过；仅使用 CPU

## 可选依赖

- `torch` extra 仅用于演示置信度模型 (`hypofuse.neural`)，且仅使用 CPU。不附带、不下载任何预训练权重。

## 模块列表

除核心模块外，还包括：

- `hypofuse.itn` -- 逆文本归一化
- `hypofuse.timings` -- per-token 时间戳
- `hypofuse.config` -- 配置序列化
- `hypofuse.runmeta` -- 运行时元数据
- `hypofuse.convert` -- manifest 转换
- `hypofuse.neural` -- 可选 torch 模型

## 开发目标

| 命令 | 说明 |
|---|---|
| `make examples` | 运行所有示例 |
| `make check-package` | 构建并验证 wheel |

完整文档索引见 [`docs/README.md`](docs/README.md)：API 参考、CLI、schema、指标定义、融合、语言模型、校准、错误分析、合成数据、可复现性与参考资料。

## 时间线说明

本项目于 2026 年 9 月创建和验证。Git 中 2025 年至 2026 年 8 月的日期是本次生成的演示时间线，不代表那些日期已经开展的工作。每次开发提交均执行构建、测试与格式检查；未使用空提交。项目未宣称论文成果、预训练模型或真实语音数据集成绩。

## 局限性

- 未在真实 ASR 系统上评估；所有测试和演示使用合成数据
- 不附带任何预训练模型权重；可选的 `torch` extra 仅用于演示性置信度模型（CPU）
- N-gram 语言模型（Katz backoff + Jelinek-Mercer）是教学级别的最小实现，不适合生产环境大规模语料
- ARPA 格式导入导出覆盖基本字段，不支持所有扩展格式
- 置信度校准在合成分布上验证，不声称在真实 ASR 假设上的校准效果
- 项目中的任何数字均不声称代表真实基准测试结果

## 许可证

MIT License -- 详见 `LICENSE` 文件。

## 参考文献

- Fiscus, J. G. (1997). A Post-Processing System to Yield Reduced Word Error Rates: Recognizer Output Voting Error Reduction (ROVER). *IEEE Workshop on Automatic Speech Recognition and Understanding*.
- Mangu, L., Brill, E., Stolcke, A. (2000). Finding Consensus in Speech Recognition. https://arxiv.org/abs/1904.08295
- Jurafsky, D., Martin, J. H. *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/ -- N-gram 与平滑章节
- jiwer -- WER/CER 指标 API 约定参考. https://github.com/jiwer/jiwer
- CMU Sphinx / KenLM -- ARPA 语言模型格式文档
