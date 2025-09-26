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
    band: int | None = None,
) -> Alignment:
    """Compute the edit alignment between two token sequences.

    Returns an :class:`Alignment` with one :class:`AlignmentOp` per cell in
    the optimal path, walking from start to end of both sequences.

    When *costs* is provided, the DP uses the specified substitution,
    insertion and deletion weights instead of the default unit cost.

    When *band* is not ``None``, the DP is restricted to cells where
    ``|i - j| <= band`` (Sakoe-Chiba band). This is an approximation
    when the band is narrower than the optimal path width. Raises
    :class:`AlignmentError` when the band is too narrow to connect
    the start and end (``abs(n - m) > band``).
    """
    if costs is not None:
        costs.validate()
    if band is not None and band < 0:
        raise AlignmentError("band must be non-negative")
    ref = _safe_tokens(reference)
    hyp = _safe_tokens(hypothesis)
    n, m = len(ref), len(hyp)
    if n == 0 and m == 0:
        return Alignment(ops=(), score=0)
    if band is not None and abs(n - m) > band:
        raise AlignmentError(f"band={band} too narrow for lengths {n} and {m}")
    c_ins = costs.insertion if costs else 1.0
    c_del = costs.deletion if costs else 1.0
    _INF = float("inf")
    dp: list[list[float]] = [[_INF] * (m + 1) for _ in range(n + 1)]
    bp: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        if band is None or i <= band:
            dp[i][0] = dp[i - 1][0] + c_del
            bp[i][0] = DEL
    for j in range(1, m + 1):
        if band is None or j <= band:
            dp[0][j] = dp[0][j - 1] + c_ins
            bp[0][j] = INS
    for i in range(1, n + 1):
        j_lo = max(1, i - band) if band is not None else 1
        j_hi = min(m, i + band) if band is not None else m
        for j in range(j_lo, j_hi + 1):
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


@dataclass(frozen=True)
class ErrorBreakdown:
    """Counts of each error type from an alignment."""

    substitutions: int
    insertions: int
    deletions: int
    total: int
    ref_length: int
    hyp_length: int


def error_breakdown(alignment: Alignment) -> ErrorBreakdown:
    """Return an :class:`ErrorBreakdown` for the given alignment."""
    subs = sum(1 for op in alignment.ops if op.op == SUB)
    ins = sum(1 for op in alignment.ops if op.op == INS)
    dels = sum(1 for op in alignment.ops if op.op == DEL)
    return ErrorBreakdown(
        substitutions=subs,
        insertions=ins,
        deletions=dels,
        total=subs + ins + dels,
        ref_length=alignment.ref_length,
        hyp_length=alignment.hyp_length,
    )


@dataclass(frozen=True)
class CorpusRates:
    """Aggregate error rates over a corpus of utterances.

    ``micro`` is total errors / total reference length.
    ``macro`` is the unweighted mean of per-utterance rates.
    """

    micro: float
    macro: float
    n: int


def corpus_error_rates(
    pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
    level: str = "word",
) -> CorpusRates:
    """Compute micro and macro error rates over a corpus.

    *pairs* is a sequence of ``(reference, hypothesis)`` pairs.
    For ``level="word"`` each element is a sequence of word
    tokens; for ``level="char"`` each element is a string.

    Raises ``ValueError`` on empty input.
    """
    if not pairs:
        raise ValueError("at least one pair is required")
    if level not in ("word", "char"):
        raise ValueError(f"unknown level: {level!r}")
    total_errors = 0
    total_ref_len = 0
    per_utterance_rates: list[float] = []
    for ref, hyp in pairs:
        align = edit_alignment(list(ref), list(hyp))
        total_errors += align.errors
        total_ref_len += align.ref_length
        if align.ref_length == 0:
            rate = 0.0 if align.hyp_length == 0 else 1.0
        else:
            rate = align.errors / align.ref_length
        per_utterance_rates.append(rate)
    micro = total_errors / total_ref_len if total_ref_len > 0 else 0.0
    macro = sum(per_utterance_rates) / len(per_utterance_rates)
    return CorpusRates(micro=micro, macro=macro, n=len(pairs))


def symmetric_error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    """Compute symmetric error rate: errors / mean length.

    The denominator is ``mean(ref_length, hyp_length)``. The
    result is in ``[0, 2]``: zero when the sequences are
    identical, approaching two when one sequence is empty and
    the other is long.
    """
    align = edit_alignment(list(reference), list(hypothesis))
    mean_len = (align.ref_length + align.hyp_length) / 2
    if mean_len == 0:
        return 0.0
    return align.errors / mean_len


def alignment_to_jsonl_row(
    alignment: Alignment,
    utterance_id: str,
    system: str = "",
) -> dict[str, object]:
    """Serialize an alignment to a JSONL-compatible dict.

    The row includes a ``schema`` and ``schema_version`` field
    for forward compatibility.
    """
    return {
        "schema": "hypofuse.alignment",
        "schema_version": 1,
        "utterance_id": utterance_id,
        "system": system,
        "score": alignment.score,
        "ref_length": alignment.ref_length,
        "hyp_length": alignment.hyp_length,
        "errors": alignment.errors,
        "ops": alignment.to_dict(),
    }


def alignment_from_jsonl_row(row: dict[str, object]) -> Alignment:
    """Deserialize an alignment from a JSONL-compatible dict.

    Raises ``ValueError`` for unknown schema, wrong version, or
    missing required fields.
    """
    schema = row.get("schema")
    if schema != "hypofuse.alignment":
        raise ValueError(f"unknown schema: {schema!r}")
    version = row.get("schema_version")
    if version != 1:
        raise ValueError(f"unsupported schema_version: {version!r}")
    required = (
        "utterance_id",
        "score",
        "ref_length",
        "hyp_length",
        "errors",
        "ops",
    )
    for key in required:
        if key not in row:
            raise ValueError(f"missing required field: {key!r}")
    ops_data = row["ops"]
    if not isinstance(ops_data, list):
        raise ValueError("ops must be a list")
    ops = tuple(AlignmentOp(op=d["op"], ref_token=d["ref"], hyp_token=d["hyp"]) for d in ops_data)
    return Alignment(ops=ops, score=float(row["score"]))
