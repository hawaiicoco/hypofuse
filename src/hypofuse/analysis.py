"""Error analysis: slice-based metrics, confusion mining, paired bootstrap.

Produces Markdown / JSONL reports. The bootstrap CI uses a deterministic
seed so reports are reproducible.
"""

from __future__ import annotations

import html
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from hypofuse.alignment import (
    DEL,
    INS,
    SUB,
    AlignmentOp,
    character_error_rate,
    edit_alignment,
    word_error_rate,
)
from hypofuse.util import seeded


@dataclass(frozen=True)
class UtteranceScore:
    utterance_id: str
    reference: tuple[str, ...]
    hypothesis: tuple[str, ...]
    speaker_group: str = ""
    noise_db: float = 0.0
    duration_s: float = 0.0
    intent_domain: str = ""

    def error_breakdown(self) -> tuple[int, int, int]:
        """Return (substitutions, deletions, insertions) from the stored alignment ops."""
        ops = edit_alignment(self.reference, self.hypothesis).ops
        subs = sum(1 for op in ops if op.op == SUB)
        dels = sum(1 for op in ops if op.op == DEL)
        ins = sum(1 for op in ops if op.op == INS)
        return (subs, dels, ins)

    @property
    def is_perfect(self) -> bool:
        """True when reference and hypothesis tokens are identical."""
        return self.reference == self.hypothesis


@dataclass(frozen=True)
class SliceMetric:
    key: str
    count: int
    cer: float
    wer: float


@dataclass(frozen=True)
class ComparisonRow:
    system_a: str
    system_b: str
    delta_cer: float
    delta_wer: float
    cer_ci_low: float
    cer_ci_high: float
    wer_ci_low: float
    wer_ci_high: float
    n_pairs: int


def _safe_tokens(seq: Sequence[str]) -> tuple[str, ...]:
    return tuple(seq)


def slice_by_duration(
    items: Iterable[UtteranceScore],
    boundaries: Sequence[float] = (1.0, 3.0, 10.0),
) -> dict[str, list[UtteranceScore]]:
    out: dict[str, list[UtteranceScore]] = defaultdict(list)
    out["all"] = []
    for item in items:
        out["all"].append(item)
        bucket = ">10s"
        for b in boundaries:
            if item.duration_s < b:
                bucket = f"<{b:g}s"
                break
        out[bucket].append(item)
    return dict(out)


def slice_by_field(
    items: Iterable[UtteranceScore], field_name: str
) -> dict[str, list[UtteranceScore]]:
    out: dict[str, list[UtteranceScore]] = defaultdict(list)
    for item in items:
        if field_name == "speaker_group":
            out[item.speaker_group or "unknown"].append(item)
        elif field_name == "intent_domain":
            out[item.intent_domain or "unknown"].append(item)
        elif field_name == "noise":
            if item.noise_db < 0:
                out["quiet"].append(item)
            elif item.noise_db < 20:
                out["moderate"].append(item)
            else:
                out["noisy"].append(item)
        else:
            raise ValueError(f"unknown field: {field_name}")
    return dict(out)


def slice_metrics(
    slices: dict[str, list[UtteranceScore]], use_cer: bool = False
) -> list[SliceMetric]:
    rows: list[SliceMetric] = []
    for key, items in slices.items():
        if not items:
            rows.append(SliceMetric(key, 0, 0.0, 0.0))
            continue
        cer = statistics.fmean(
            character_error_rate(
                "".join(_safe_tokens(it.reference)), "".join(_safe_tokens(it.hypothesis))
            )
            for it in items
        )
        wer = statistics.fmean(
            word_error_rate(list(it.reference), list(it.hypothesis)) for it in items
        )
        if not use_cer:
            cer = 0.0
        rows.append(SliceMetric(key, len(items), cer, wer))
    return rows


def substitution_pairs(
    items: Iterable[UtteranceScore],
    *,
    top_n: int | None = None,
    directional: bool = True,
    min_count: int = 1,
    level: str = "word",
) -> Counter[tuple[str, str]]:
    """Mine confused token pairs from edit alignment.

    By default every substitution is counted in reference-to-hypothesis
    order.  Set ``directional=False`` to collapse (a, b) and (b, a) into
    a single sorted pair.  ``min_count`` drops pairs below the threshold
    and ``top_n`` keeps only the *n* most frequent pairs (ties broken
    by count descending, then lexicographic on the key tuple).
    ``level="char"`` joins all tokens into a character sequence before
    alignment, which is useful for CJK languages.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for item in items:
        if level == "char":
            ref_seq = tuple("".join(item.reference))
            hyp_seq = tuple("".join(item.hypothesis))
        else:
            ref_seq = item.reference
            hyp_seq = item.hypothesis
        ops: list[AlignmentOp] = edit_alignment(ref_seq, hyp_seq).ops
        for op in ops:
            if op.op == SUB:
                ref_tok = str(op.ref_token)
                hyp_tok = str(op.hyp_token)
                if directional:
                    counts[(ref_tok, hyp_tok)] += 1
                else:
                    pair = tuple(sorted((ref_tok, hyp_tok)))
                    counts[pair] += 1
    if min_count > 1:
        counts = Counter({k: v for k, v in counts.items() if v >= min_count})
    if top_n is not None:
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        counts = Counter(dict(ranked[:top_n]))
    return counts


def paired_bootstrap_ci(
    system_a: Sequence[UtteranceScore],
    system_b: Sequence[UtteranceScore],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ComparisonRow:
    if len(system_a) != len(system_b):
        raise ValueError("system lists must have the same length")
    rng = random.Random(seed)
    n = len(system_a)
    deltas_cer: list[float] = []
    deltas_wer: list[float] = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        a = [system_a[i] for i in idxs]
        b = [system_b[i] for i in idxs]
        cer_a = statistics.fmean(
            character_error_rate("".join(it.reference), "".join(it.hypothesis)) for it in a
        )
        cer_b = statistics.fmean(
            character_error_rate("".join(it.reference), "".join(it.hypothesis)) for it in b
        )
        wer_a = statistics.fmean(
            word_error_rate(list(it.reference), list(it.hypothesis)) for it in a
        )
        wer_b = statistics.fmean(
            word_error_rate(list(it.reference), list(it.hypothesis)) for it in b
        )
        deltas_cer.append(cer_b - cer_a)
        deltas_wer.append(wer_b - wer_a)
    alpha = (1 - confidence) / 2
    lo_c, hi_c = _quantile(deltas_cer, alpha), _quantile(deltas_cer, 1 - alpha)
    lo_w, hi_w = _quantile(deltas_wer, alpha), _quantile(deltas_wer, 1 - alpha)
    cer_a_all = statistics.fmean(
        character_error_rate("".join(it.reference), "".join(it.hypothesis)) for it in system_a
    )
    cer_b_all = statistics.fmean(
        character_error_rate("".join(it.reference), "".join(it.hypothesis)) for it in system_b
    )
    wer_a_all = statistics.fmean(
        word_error_rate(list(it.reference), list(it.hypothesis)) for it in system_a
    )
    wer_b_all = statistics.fmean(
        word_error_rate(list(it.reference), list(it.hypothesis)) for it in system_b
    )
    return ComparisonRow(
        system_a="A",
        system_b="B",
        delta_cer=cer_b_all - cer_a_all,
        delta_wer=wer_b_all - wer_a_all,
        cer_ci_low=lo_c,
        cer_ci_high=hi_c,
        wer_ci_low=lo_w,
        wer_ci_high=hi_w,
        n_pairs=n,
    )


def _quantile(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _escape_markdown(text: str) -> str:
    """Escape pipe, backslash and newline for safe markdown table cells."""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n")


def report_to_markdown(
    slices: list[SliceMetric],
    comparisons: Sequence[ComparisonRow],
    *,
    meta: Mapping[str, object] | None = None,
) -> str:
    """Render an error-analysis report as Markdown.

    When *meta* is given, a ``## Run`` section with sorted key-value
    lines is inserted before the slices table.
    """
    lines: list[str] = ["# Error Analysis Report", ""]
    if meta is not None:
        lines.append("## Run")
        for key in sorted(meta):
            lines.append(f"{key}: {meta[key]}")
        lines.append("")
    lines.append("## Slices")
    lines.append("| Slice | Count | CER | WER |")
    lines.append("|---|---|---|---|")
    for s in slices:
        key = _escape_markdown(html.escape(s.key))
        lines.append(f"| {key} | {s.count} | {s.cer:.4f} | {s.wer:.4f} |")
    lines.append("")
    lines.append("## System comparisons (paired bootstrap, 95% CI)")
    lines.append("| Pair | n | Δ CER | CER CI | Δ WER | WER CI |")
    lines.append("|---|---|---|---|---|---|")
    for c in comparisons:
        cer_ci = f"[{c.cer_ci_low:.4f}, {c.cer_ci_high:.4f}]"
        wer_ci = f"[{c.wer_ci_low:.4f}, {c.wer_ci_high:.4f}]"
        pair = _escape_markdown(f"{c.system_a} vs {c.system_b}")
        lines.append(
            f"| {pair} | {c.n_pairs} | "
            f"{c.delta_cer:.4f} | {cer_ci} | {c.delta_wer:.4f} | {wer_ci} |"
        )
    return "\n".join(lines)


def report_to_jsonl(slices: list[SliceMetric], comparisons: Sequence[ComparisonRow]) -> str:
    """Render the report as JSON Lines, one record per slice or comparison."""
    import json

    lines: list[str] = []
    for s in slices:
        lines.append(
            json.dumps(
                {
                    "kind": "slice",
                    "schema": "hypofuse.report",
                    "schema_version": 1,
                    "key": s.key,
                    "count": s.count,
                    "cer": s.cer,
                    "wer": s.wer,
                },
                sort_keys=True,
            )
        )
    for c in comparisons:
        lines.append(
            json.dumps(
                {
                    "kind": "comparison",
                    "schema": "hypofuse.report",
                    "schema_version": 1,
                    "system_a": c.system_a,
                    "system_b": c.system_b,
                    "delta_cer": c.delta_cer,
                    "delta_wer": c.delta_wer,
                    "cer_ci": [c.cer_ci_low, c.cer_ci_high],
                    "wer_ci": [c.wer_ci_low, c.wer_ci_high],
                    "n_pairs": c.n_pairs,
                },
                sort_keys=True,
            )
        )
    return "\n".join(lines)


def corpus_error_rate(scores: Iterable[UtteranceScore], average: str = "micro") -> float:
    """Aggregate error rate over a corpus.

    ``average="micro"`` computes total errors / total reference tokens
    (the corpus-level rate).  ``average="macro"`` computes the arithmetic
    mean of per-utterance error rates.  These differ when utterances
    have different reference lengths: a short utterance with many errors
    contributes more to the macro average than to the micro average.
    """
    if average not in ("micro", "macro"):
        raise ValueError(f"unknown average: {average!r}; use 'micro' or 'macro'")
    items = list(scores)
    if not items:
        return 0.0
    if average == "micro":
        total_errors = 0
        total_ref = 0
        for it in items:
            align = edit_alignment(it.reference, it.hypothesis)
            total_errors += align.errors
            total_ref += align.ref_length
        if total_ref == 0:
            return 0.0
        return total_errors / total_ref
    if average == "macro":
        rates = [word_error_rate(list(it.reference), list(it.hypothesis)) for it in items]
        return statistics.fmean(rates)
    raise ValueError(f"unknown average: {average!r}; use 'micro' or 'macro'")


def paired_bootstrap_p_value(
    a_scores: Sequence[UtteranceScore],
    b_scores: Sequence[UtteranceScore],
    *,
    seed: int = 0,
    iterations: int = 1000,
) -> float:
    """One-sided p-value that system A is not better than system B.

    Resamples utterance indices with replacement ``iterations`` times,
    counts how often the mean WER difference (B - A) is <= 0 (i.e. A's
    advantage does not hold in the resample), and applies add-one
    smoothing::

        p = (count_le + 1) / (iterations + 1)

    A small p-value indicates A is reliably better.  Deterministic
    via ``hypofuse.util.seeded``.  Raises ``ValueError`` when
    ``iterations < 1`` or the score lists differ in length.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if len(a_scores) != len(b_scores):
        raise ValueError("score lists must have the same length")
    rng = seeded(seed)
    n = len(a_scores)
    wer_a = [word_error_rate(list(it.reference), list(it.hypothesis)) for it in a_scores]
    wer_b = [word_error_rate(list(it.reference), list(it.hypothesis)) for it in b_scores]
    count_le = 0
    for _ in range(iterations):
        idxs = [rng.randrange(n) for _ in range(n)]
        mean_diff = sum(wer_b[i] - wer_a[i] for i in idxs) / n
        if mean_diff <= 0:
            count_le += 1
    return (count_le + 1) / (iterations + 1)


def slice_by_quantiles(
    scores: Iterable[UtteranceScore],
    field: str,
    *,
    buckets: int = 4,
) -> dict[str, list[UtteranceScore]]:
    """Partition *scores* into quantile buckets using nearest-rank edges.

    Edges are derived from the data without scipy.  If all field values
    are equal, a single ``"all"`` slice is returned.  Each score lands in
    exactly one bucket.  Raises ``ValueError`` when ``buckets < 1``.
    """
    if buckets < 1:
        raise ValueError("buckets must be >= 1")
    items = list(scores)
    if not items:
        return {}
    values = [getattr(it, field) for it in items]
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    # Nearest-rank edges at evenly spaced positions
    raw_edges: list[float] = []
    for i in range(buckets + 1):
        idx = min(i * n // buckets, n - 1)
        raw_edges.append(sorted_vals[idx])

    # Deduplicate while preserving order
    unique_edges: list[float] = []
    for e in raw_edges:
        if not unique_edges or e != unique_edges[-1]:
            unique_edges.append(e)

    if len(unique_edges) <= 1:
        return {"all": items}

    # Build interval labels
    labels = [f"q{i}" for i in range(len(unique_edges) - 1)]
    out: dict[str, list[UtteranceScore]] = {lab: [] for lab in labels}

    # Assign each item: [lo, hi) for all but last; [lo, hi] for last
    for item, val in zip(items, values, strict=True):
        for i, lab in enumerate(labels):
            lo = unique_edges[i]
            hi = unique_edges[i + 1]
            in_interval = lo <= val <= hi if i == len(labels) - 1 else lo <= val < hi
            if in_interval:
                out[lab].append(item)
                break

    return out


@dataclass(frozen=True)
class GroupBiasRow:
    """One row in a group bias table."""

    group: str
    count: int
    wer: float
    wer_ci_low: float
    wer_ci_high: float


def group_bias_table(
    scores: Sequence[UtteranceScore],
    field: str,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[GroupBiasRow]:
    """Per-group error rate with bootstrap CI, sorted by WER descending.

    Groups are formed by slicing *scores* on *field* (same fields as
    :func:`slice_by_field`).  The CI uses percentile bootstrap over
    utterance-level WER within each group.  Rows are sorted by WER
    descending, then by group name for deterministic ordering.
    """
    slices = slice_by_field(list(scores), field)
    rng = seeded(seed)
    rows: list[GroupBiasRow] = []
    for key, items in slices.items():
        if not items:
            rows.append(GroupBiasRow(key, 0, 0.0, 0.0, 0.0))
            continue
        wers = [word_error_rate(list(it.reference), list(it.hypothesis)) for it in items]
        point_wer = statistics.fmean(wers)
        bootstrap_means: list[float] = []
        n = len(wers)
        for _ in range(n_bootstrap):
            sample = [wers[rng.randrange(n)] for _ in range(n)]
            bootstrap_means.append(statistics.fmean(sample))
        alpha = (1 - confidence) / 2
        ci_lo = _quantile(bootstrap_means, alpha)
        ci_hi = _quantile(bootstrap_means, 1 - alpha)
        rows.append(GroupBiasRow(key, len(items), point_wer, ci_lo, ci_hi))
    rows.sort(key=lambda r: (-r.wer, r.group))
    return rows
