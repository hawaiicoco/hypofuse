"""ROVER-style consensus fusion over an alignment grid.

The :func:`fuse` entry point takes the output of :func:`multi_align.progressive_align`
together with optional per-token scores from each hypothesis, applies a voting
policy to pick a token per grid column, and emits the fused token sequence with
per-token fused confidences.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from hypofuse.exceptions import FusionError
from hypofuse.multi_align import GAP, TokenGrid
from hypofuse.util import seeded

POLICY_MAJORITY = "majority"
POLICY_SCORE_WEIGHTED = "score_weighted"
POLICY_LM_WEIGHTED = "lm_weighted"
POLICY_POSTERIOR_WEIGHTED = "posterior_weighted"
POLICY_CONFIDENCE_WEIGHTED = "confidence_weighted"

_VALID_POLICIES = {
    POLICY_MAJORITY,
    POLICY_SCORE_WEIGHTED,
    POLICY_LM_WEIGHTED,
    POLICY_POSTERIOR_WEIGHTED,
    POLICY_CONFIDENCE_WEIGHTED,
}


@dataclass(frozen=True)
class FusionConfig:
    policy: str = POLICY_MAJORITY
    alpha: float = 1.0  # weight for LM scores when policy is LM-weighted
    beta: float = 1.0  # weight for acoustic scores when policy is score-weighted
    tie_break: str = "lexicographic"  # "lexicographic" | "first"
    null_token: str = GAP  # used for gap columns
    null_policy: str = "keep"  # "keep"|"drop"|"vote"; default matches current behaviour
    seed: int = 0  # used by tie_break="seeded"

    def validate(self) -> None:
        if self.policy not in _VALID_POLICIES:
            raise FusionError(f"unknown policy: {self.policy!r}")
        if self.alpha < 0 or self.beta < 0:
            raise FusionError("alpha and beta must be non-negative")
        if self.tie_break not in {"lexicographic", "first", "highest_score", "seeded"}:
            raise FusionError(f"unknown tie_break: {self.tie_break!r}")
        if self.null_policy not in {"keep", "drop", "vote"}:
            raise FusionError(f"unknown null_policy: {self.null_policy!r}")


@dataclass(frozen=True)
class FusionResult:
    tokens: tuple[str, ...]
    confidences: tuple[float, ...]
    chosen: tuple[tuple[Hashable, float], ...]  # (token, score) per column
    agreements: tuple[float, ...] = ()
    margins: tuple[float, ...] = ()
    entropies: tuple[float, ...] = ()


def _majority_vote(
    candidates: list[tuple[Hashable, float]],
    config: FusionConfig,
) -> tuple[Hashable, float]:
    counts: dict[Hashable, float] = {}
    for token, weight in candidates:
        counts[token] = counts.get(token, 0.0) + weight
    top_score = max(counts.values())
    winners = [tok for tok, score in counts.items() if score == top_score]
    chosen = _tie_break(winners, config, candidates)
    return chosen, top_score


def _score_weighted_vote(
    candidates: list[tuple[Hashable, float]],
    config: FusionConfig,
) -> tuple[Hashable, float]:
    return _majority_vote(candidates, config)


def _lm_weighted_vote(
    candidates: list[tuple[Hashable, float]],
    lm_scores: list[float],
    config: FusionConfig,
) -> tuple[Hashable, float]:
    weighted: list[tuple[Hashable, float]] = []
    for (token, score), lm in zip(candidates, lm_scores, strict=False):
        weighted.append((token, score + config.alpha * lm))
    return _majority_vote(weighted, config)


def _posterior_weighted_vote(
    candidates: list[tuple[Hashable, float]],
    posteriors_col: list[float],
    config: FusionConfig,
) -> tuple[Hashable, float]:
    """Weight each candidate by its per-token posterior, then majority vote."""
    weighted: list[tuple[Hashable, float]] = []
    for (token, weight), post in zip(candidates, posteriors_col, strict=False):
        weighted.append((token, weight * max(0.0, post)))
    return _majority_vote(weighted, config)


def _confidence_weighted_vote(
    candidates: list[tuple[Hashable, float]],
    hyp_weights: list[float],
    config: FusionConfig,
) -> tuple[Hashable, float]:
    """Weight each candidate by its hypothesis-level weight, then majority vote.

    Unlike ``score_weighted`` (which reads per-token acoustic scores),
    this policy uses one scalar weight per hypothesis.
    """
    weighted: list[tuple[Hashable, float]] = []
    for (token, score), hw in zip(candidates, hyp_weights, strict=False):
        weighted.append((token, score * max(0.0, hw)))
    return _majority_vote(weighted, config)


def _tie_break(
    winners: Sequence[Hashable],
    config: FusionConfig,
    candidates: list[tuple[Hashable, float]] | None = None,
) -> Hashable:
    if config.tie_break == "first":
        return winners[0]
    if config.tie_break == "highest_score" and candidates is not None:
        winner_set = set(winners)
        best: dict[Hashable, float] = {}
        for tok, w in candidates:
            if tok in winner_set:
                best[tok] = max(best.get(tok, -float("inf")), w)
        return max(best, key=lambda t: best[t])
    if config.tie_break == "seeded":
        rng = seeded(config.seed)
        return rng.choice(list(winners))
    return sorted(winners, key=lambda x: str(x))[0]


def fuse(
    grid: TokenGrid,
    scores: Sequence[Sequence[float]] | None = None,
    lm_scores: Sequence[Sequence[float]] | None = None,
    config: FusionConfig | None = None,
    posteriors: Sequence[Sequence[float]] | None = None,
    weights: Sequence[float] | None = None,
) -> FusionResult:
    """Fuse hypotheses aligned on ``grid`` into a single token sequence.

    ``scores`` and ``lm_scores`` are optional per-token, per-hypothesis
    numeric weights (acoustic log10 / LM log10 by convention). Missing
    scores are treated as 0.

    ``posteriors`` is an optional per-token, per-hypothesis posterior
    probability grid used by the ``posterior_weighted`` policy. Must
    have one row per hypothesis and one column per grid position.
    """
    config = config or FusionConfig()
    config.validate()
    n_hyps = grid.depth
    score_grid: list[list[float]] = [[0.0] * n_hyps for _ in range(grid.width)]
    lm_grid: list[list[float]] = [[0.0] * n_hyps for _ in range(grid.width)]
    if scores is not None:
        for h_idx, row in enumerate(scores):
            for col_idx in range(grid.width):
                if h_idx < n_hyps and col_idx < len(row):
                    score_grid[col_idx][h_idx] = float(row[col_idx])
    if lm_scores is not None:
        for h_idx, row in enumerate(lm_scores):
            for col_idx in range(grid.width):
                if h_idx < n_hyps and col_idx < len(row):
                    lm_grid[col_idx][h_idx] = float(row[col_idx])
    post_grid: list[list[float]] = [[1.0] * n_hyps for _ in range(grid.width)]
    if posteriors is not None:
        if len(posteriors) != n_hyps:
            raise FusionError(f"posteriors rows ({len(posteriors)}) != grid depth ({n_hyps})")
        for h_idx, row in enumerate(posteriors):
            if len(row) != grid.width:
                raise FusionError(
                    f"posteriors row {h_idx} length ({len(row)}) != grid width ({grid.width})"
                )
            for col_idx in range(grid.width):
                post_grid[col_idx][h_idx] = float(row[col_idx])
    if config.policy == POLICY_POSTERIOR_WEIGHTED and posteriors is None:
        raise FusionError("posterior_weighted policy requires posteriors")
    conf_weights: list[float] = [1.0] * n_hyps
    if weights is not None:
        if len(weights) != n_hyps:
            raise FusionError(f"weights length ({len(weights)}) != grid depth ({n_hyps})")
        conf_weights = [float(w) for w in weights]
    tokens: list[str] = []
    confidences: list[float] = []
    chosen: list[tuple[Hashable, float]] = []
    agreements_list: list[float] = []
    margins_list: list[float] = []
    entropies_list: list[float] = []
    for col_idx, column in enumerate(grid.columns):
        candidates: list[tuple[Hashable, float]] = []
        lm_col: list[float] = []
        post_col: list[float] = []
        weights_col: list[float] = []
        for h_idx, token in enumerate(column):
            if token == GAP and config.null_policy != "vote":
                continue
            weight = score_grid[col_idx][h_idx]
            weight = 1.0 if scores is None else max(0.0, weight)
            candidates.append((token, weight))
            lm_col.append(lm_grid[col_idx][h_idx])
            post_col.append(post_grid[col_idx][h_idx])
            weights_col.append(conf_weights[h_idx])
        if not candidates:
            if config.null_policy == "drop":
                continue
            tokens.append(config.null_token)
            confidences.append(0.0)
            chosen.append((config.null_token, 0.0))
            agreements_list.append(0.0)
            margins_list.append(0.0)
            entropies_list.append(0.0)
            continue
        if config.policy == POLICY_MAJORITY:
            tok, score = _majority_vote(candidates, config)
        elif config.policy == POLICY_SCORE_WEIGHTED:
            tok, score = _score_weighted_vote(candidates, config)
        elif config.policy == POLICY_LM_WEIGHTED:
            tok, score = _lm_weighted_vote(candidates, lm_col, config)
        elif config.policy == POLICY_POSTERIOR_WEIGHTED:
            tok, score = _posterior_weighted_vote(candidates, post_col, config)
        elif config.policy == POLICY_CONFIDENCE_WEIGHTED:
            tok, score = _confidence_weighted_vote(candidates, weights_col, config)
        else:
            raise FusionError(f"unreachable policy: {config.policy}")
        agreement_count = sum(1 for c, _ in candidates if c == tok) / len(candidates)
        # Mass-based statistics
        mass_counts: dict[Hashable, float] = {}
        for c_tok, c_w in candidates:
            mass_counts[c_tok] = mass_counts.get(c_tok, 0.0) + c_w
        total_mass = sum(mass_counts.values())
        sorted_masses = sorted(mass_counts.values(), reverse=True)
        winner_mass = mass_counts.get(tok, 0.0)
        runner_up_mass = sorted_masses[1] if len(sorted_masses) > 1 else 0.0
        agg = winner_mass / total_mass if total_mass > 0 else 0.0
        marg = (winner_mass - runner_up_mass) / total_mass if total_mass > 0 else 0.0
        ent = 0.0
        if total_mass > 0:
            for mass_val in sorted_masses:
                p = mass_val / total_mass
                if p > 0:
                    ent -= p * math.log(p)
        tokens.append(str(tok))
        confidences.append(float(agreement_count))
        chosen.append((tok, score))
        agreements_list.append(float(agg))
        margins_list.append(float(marg))
        entropies_list.append(float(ent))
    return FusionResult(
        tokens=tuple(tokens),
        confidences=tuple(confidences),
        chosen=tuple(chosen),
        agreements=tuple(agreements_list),
        margins=tuple(margins_list),
        entropies=tuple(entropies_list),
    )


def fusion_invariants(result: FusionResult, inputs: Sequence[Sequence[str]]) -> bool:
    """The fused token set is a subset of the input token sets, never new ones."""
    fused = set(result.tokens)
    universe: set[str] = set()
    for seq in inputs:
        universe.update(seq)
    return fused.issubset(universe)
