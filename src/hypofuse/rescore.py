"""N-best rescoring with a shallow-fusion-style LM.

Given a list of hypotheses with acoustic log10 scores and an N-gram LM,
``rescore`` produces new scores by combining the two with a configurable
LM weight. ``lm_weight_sweep`` evaluates a grid of weights and reports the
resulting metrics in a small table.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from hypofuse.alignment import character_error_rate, word_error_rate
from hypofuse.ngram import EOS, NgramLM

LM_LOG_BASE = 10.0


@dataclass(frozen=True)
class ScoredHypothesis:
    text: str
    tokens: tuple[str, ...]
    acoustic_log10: float
    lm_log10: float

    @classmethod
    def from_tokens(cls, tokens: Sequence[str], acoustic_log10: float = 0.0) -> ScoredHypothesis:
        text = " ".join(tokens)
        return cls(text=text, tokens=tuple(tokens), acoustic_log10=acoustic_log10, lm_log10=0.0)


def rescore_lm(hyp: ScoredHypothesis, lm: NgramLM) -> float:
    """Compute the LM log10 score of a hypothesis using the given model."""
    if not hyp.tokens:
        return 0.0
    sequence = [EOS] * (lm.order - 1) + list(hyp.tokens) + [EOS]
    score = 0.0
    for i in range(lm.order - 1, len(sequence)):
        ng = tuple(sequence[i - lm.order + 1 : i + 1])
        p = max(lm.prob_katz(ng), 1e-12)
        score += math.log10(p)
    return score


def shallow_fusion_score(
    hyp: ScoredHypothesis,
    lm: NgramLM,
    lm_weight: float,
    acoustic_weight: float = 1.0,
    insertion_penalty: float = 0.0,
) -> float:
    """Combine acoustic log10 + LM-weight * LM log10 into a single score."""
    lm_score = rescore_lm(hyp, lm)
    return (
        acoustic_weight * hyp.acoustic_log10
        + lm_weight * lm_score
        - insertion_penalty * len(hyp.tokens)
    )


def rescore_nbest(
    nbest: Sequence[ScoredHypothesis],
    lm: NgramLM,
    lm_weight: float,
    acoustic_weight: float = 1.0,
) -> tuple[ScoredHypothesis, ...]:
    """Re-rank an n-best list with shallow fusion and return the new order."""
    scored = [
        ScoredHypothesis(
            text=h.text,
            tokens=h.tokens,
            acoustic_log10=h.acoustic_log10,
            lm_log10=rescore_lm(h, lm),
        )
        for h in nbest
    ]
    ranked = sorted(
        scored,
        key=lambda h: -(acoustic_weight * h.acoustic_log10 + lm_weight * h.lm_log10),
    )
    return tuple(ranked)


def lm_weight_sweep(
    nbest: Sequence[ScoredHypothesis],
    reference: Sequence[str],
    lm: NgramLM,
    weights: Sequence[float] = (0.0, 0.1, 0.2, 0.5, 1.0),
) -> list[tuple[float, float, str]]:
    """Return [(weight, CER-or-WER, chosen_text)] for each weight.

    The metric is CER if any hypothesis contains multi-character tokens, else WER.
    """
    rows: list[tuple[float, float, str]] = []
    use_cer = any(len(t) > 1 for hyp in nbest for t in hyp.tokens)
    for w in weights:
        ranked = rescore_nbest(nbest, lm, w)
        if not ranked:
            rows.append((w, float("inf"), ""))
            continue
        top = ranked[0]
        if use_cer:
            score = character_error_rate("".join(reference), top.text)
        else:
            score = word_error_rate(list(reference), top.tokens)
        rows.append((w, score, top.text))
    return rows


def lm_weight_monotonicity(
    nbest: Sequence[ScoredHypothesis],
    lm: NgramLM,
    weights: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
) -> bool:
    """Verify LM-score monotonicity for tied acoustic scores.

    For every pair (a, b), if ``a.lm_log10 >= b.lm_log10`` and
    ``a.acoustic_log10 == b.acoustic_log10``, then a must not rank
    below b at any of the given weights."""
    if not nbest:
        return True
    lm_scores = [rescore_lm(h, lm) for h in nbest]
    for i in range(len(nbest)):
        for j in range(i + 1, len(nbest)):
            a, b = nbest[i], nbest[j]
            if a.acoustic_log10 != b.acoustic_log10:
                continue
            pairs: list[tuple[int, int]] = []
            if a.lm_log10 >= b.lm_log10:
                pairs.append((i, j))
            if b.lm_log10 >= a.lm_log10:
                pairs.append((j, i))
            for idx_must, idx_other in pairs:
                for w in weights:
                    s_must = nbest[idx_must].acoustic_log10 + w * lm_scores[idx_must]
                    s_other = nbest[idx_other].acoustic_log10 + w * lm_scores[idx_other]
                    if s_must < s_other:
                        return False
    return True
