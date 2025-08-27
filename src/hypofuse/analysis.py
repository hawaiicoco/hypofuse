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
from collections.abc import Iterable, Sequence
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
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for item in items:
        ops: list[AlignmentOp] = edit_alignment(item.reference, item.hypothesis).ops
        for op in ops:
            if op.op == SUB:
                counts[(str(op.ref_token), str(op.hyp_token))] += 1
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


def report_to_markdown(slices: list[SliceMetric], comparisons: Sequence[ComparisonRow]) -> str:
    """Render an error-analysis report as Markdown."""
    lines: list[str] = ["# Error Analysis Report", ""]
    lines.append("## Slices")
    lines.append("| Slice | Count | CER | WER |")
    lines.append("|---|---|---|---|")
    for s in slices:
        lines.append(f"| {html.escape(s.key)} | {s.count} | {s.cer:.4f} | {s.wer:.4f} |")
    lines.append("")
    lines.append("## System comparisons (paired bootstrap, 95% CI)")
    lines.append("| Pair | n | Δ CER | CER CI | Δ WER | WER CI |")
    lines.append("|---|---|---|---|---|---|")
    for c in comparisons:
        cer_ci = f"[{c.cer_ci_low:.4f}, {c.cer_ci_high:.4f}]"
        wer_ci = f"[{c.wer_ci_low:.4f}, {c.wer_ci_high:.4f}]"
        lines.append(
            f"| {c.system_a} vs {c.system_b} | {c.n_pairs} | "
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
                {"kind": "slice", "key": s.key, "count": s.count, "cer": s.cer, "wer": s.wer},
                sort_keys=True,
            )
        )
    for c in comparisons:
        lines.append(
            json.dumps(
                {
                    "kind": "comparison",
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
