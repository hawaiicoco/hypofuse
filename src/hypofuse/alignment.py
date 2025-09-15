"""Edit alignment core (Levenshtein-style) for reference/hypothesis pairs.

Both word-level (WER) and character-level (CER) alignments are produced by
the same :func:`edit_alignment` function operating on token sequences. The
alignment is built with a standard dynamic-programming table; ties are
broken consistently in (substitution, insertion, deletion) order so the
output is deterministic for the same inputs.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from hypofuse.exceptions import AlignmentError

# Edit operation codes used in alignment paths.
MATCH = "MATCH"
SUB = "SUB"
INS = "INS"  # extra tokens in hypothesis
DEL = "DEL"  # tokens in reference missing from hypothesis


@dataclass(frozen=True)
class AlignmentOp:
    op: str  # one of MATCH, SUB, INS, DEL
    ref_token: Hashable | None
    hyp_token: Hashable | None

    def is_error(self) -> bool:
        return self.op != MATCH


@dataclass(frozen=True)
class AlignmentCosts:
    """Cost model for edit alignment.

    The default values reproduce unit-cost Levenshtein distance exactly.
    ``token_costs`` maps ``(ref_token, hyp_token)`` pairs to custom
    substitution costs (e.g. homophones may be cheaper). Pairs not in
    the map fall back to ``substitution``.
    """

    substitution: float = 1.0
    insertion: float = 1.0
    deletion: float = 1.0
    token_costs: Mapping[tuple[Hashable, Hashable], float] = field(default_factory=dict)

    def validate(self) -> None:
        """Reject negative or non-finite cost values."""
        for name in ("substitution", "insertion", "deletion"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {v}")
        for pair, cost in self.token_costs.items():
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(f"token_costs{pair!r} must be finite and non-negative, got {cost}")


@dataclass(frozen=True)
class Alignment:
    ops: tuple[AlignmentOp, ...]
    score: float

    @property
    def ref_length(self) -> int:
        return sum(1 for op in self.ops if op.op in {MATCH, SUB, DEL})

    @property
    def hyp_length(self) -> int:
        return sum(1 for op in self.ops if op.op in {MATCH, SUB, INS})

    @property
    def errors(self) -> int:
        return sum(1 for op in self.ops if op.is_error())

    def to_dict(self) -> list[dict[str, object]]:
        return [{"op": op.op, "ref": op.ref_token, "hyp": op.hyp_token} for op in self.ops]


def _safe_tokens(
    tokens: Sequence[Hashable] | Iterable[Hashable],
) -> tuple[Hashable, ...]:
    if isinstance(tokens, str):
        raise AlignmentError("tokens must not be a string; pass a list/tuple")
    return tuple(tokens)


def _sub_cost(
    costs: AlignmentCosts | None,
    ref_tok: Hashable,
    hyp_tok: Hashable,
) -> float:
    """Return the substitution cost for a specific token pair."""
    if costs is None:
        return 1.0
    return costs.token_costs.get((ref_tok, hyp_tok), costs.substitution)


def edit_alignment(
    reference: Sequence[Hashable],
    hypothesis: Sequence[Hashable],
    *,
    costs: AlignmentCosts | None = None,
) -> Alignment:
    """Compute the edit alignment between two token sequences.

    Returns an :class:`Alignment` with one :class:`AlignmentOp` per cell in
    the optimal path, walking from start to end of both sequences.

    When *costs* is provided, the DP uses the specified substitution,
    insertion and deletion weights instead of the default unit cost.
    """
    if costs is not None:
        costs.validate()
    ref = _safe_tokens(reference)
    hyp = _safe_tokens(hypothesis)
    n, m = len(ref), len(hyp)
    if n == 0 and m == 0:
        return Alignment(ops=(), score=0)
    c_ins = costs.insertion if costs else 1.0
    c_del = costs.deletion if costs else 1.0
    dp: list[list[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]
    bp: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + c_del
        bp[i][0] = DEL
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + c_ins
        bp[0][j] = INS
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bp[i][j] = MATCH
            else:
                sc = _sub_cost(costs, ref[i - 1], hyp[j - 1])
                sub = dp[i - 1][j - 1] + sc
                ins = dp[i][j - 1] + c_ins
                dele = dp[i - 1][j] + c_del
                best = min(sub, ins, dele)
                if best == sub:
                    dp[i][j] = sub
                    bp[i][j] = SUB
                elif best == ins:
                    dp[i][j] = ins
                    bp[i][j] = INS
                else:
                    dp[i][j] = dele
                    bp[i][j] = DEL
    ops: list[AlignmentOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bp[i][j]
        if op == MATCH:
            ops.append(AlignmentOp(MATCH, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif op == SUB:
            ops.append(AlignmentOp(SUB, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif op == INS:
            ops.append(AlignmentOp(INS, None, hyp[j - 1]))
            j -= 1
        elif op == DEL:
            ops.append(AlignmentOp(DEL, ref[i - 1], None))
            i -= 1
        else:
            raise AlignmentError(f"unreachable cell ({i},{j})")
    ops.reverse()
    return Alignment(ops=tuple(ops), score=dp[n][m])


def word_error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    """Compute word-level error rate (sub+ins+del divided by reference length)."""
    ref = list(reference)
    hyp = list(hypothesis)
    align = edit_alignment(ref, hyp)
    if align.ref_length == 0:
        return 0.0 if align.hyp_length == 0 else 1.0
    return align.errors / align.ref_length


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute character-level error rate for the two strings."""
    align = edit_alignment(list(reference), list(hypothesis))
    if align.ref_length == 0:
        return 0.0 if align.hyp_length == 0 else 1.0
    return align.errors / align.ref_length


def alignment_to_markdown(alignment: Alignment, max_rows: int = 20) -> str:
    """Render an alignment as a small markdown table for debugging."""
    lines = ["| op | ref | hyp |", "|---|---|---|"]
    for op in alignment.ops[:max_rows]:
        lines.append(f"| {op.op} | {op.ref_token!r} | {op.hyp_token!r} |")
    if len(alignment.ops) > max_rows:
        lines.append(f"| ... ({len(alignment.ops) - max_rows} more) ... | | |")
    return "\n".join(lines)
