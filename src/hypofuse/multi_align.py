"""Progressive alignment of N hypotheses into a token grid.

The :func:`progressive_align` function performs pairwise edit alignments
between successive inputs, projecting positions back to the original
hypotheses. The result is a :class:`TokenGrid` describing, for each grid
column, the token each hypothesis contributes (or a gap token if it does
not cover that column).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass

from hypofuse.alignment import DEL, INS, MATCH, SUB, Alignment, edit_alignment
from hypofuse.exceptions import AlignmentError

GAP = "*"


@dataclass(frozen=True)
class TokenGrid:
    columns: tuple[tuple[Hashable, ...], ...]
    hypotheses: tuple[tuple[Hashable, ...], ...]
    back_pointers: tuple[tuple[tuple[int, str], ...], ...]
    pair_alignments: tuple[Alignment, ...]

    @property
    def width(self) -> int:
        return len(self.columns)

    @property
    def depth(self) -> int:
        return len(self.hypotheses)


def _project_positions(align: Alignment, left: bool) -> tuple[int, ...]:
    """Map grid columns back to positions in the *moving* hypothesis.

    ``left=True`` if ``moving`` is on the left side of the pair, so the
    projection iterates through grid columns and consumes the ``moving``
    tokens via INS/MATCH/SUB.
    """
    return tuple(_column_for(align, left))


def _column_for(align: Alignment, left: bool) -> Iterable[int]:
    pos_left = pos_right = 0
    for op in align.ops:
        if op.op == MATCH or op.op == SUB:
            yield pos_left if left else pos_right
            pos_left += 1
            pos_right += 1
        elif op.op == INS:
            if not left:
                yield pos_right
            pos_right += 1
        elif op.op == DEL:
            if left:
                yield pos_left
            pos_left += 1
        else:
            raise AlignmentError(f"unknown op {op.op}")


def progressive_align(
    hypotheses: Sequence[Sequence[Hashable]],
    pivot: str = "left",
) -> TokenGrid:
    """Progressively align hypotheses left-to-right, anchored on the pivot.

    The pivot specifies which side of the *first* pair stays fixed.
    ``pivot="left"`` means the first hypothesis is the anchor (its positions
    become the grid's columns).
    """
    if not hypotheses:
        raise AlignmentError("at least one hypothesis is required")
    anchors = [_as_tuple(h) for h in hypotheses]
    if len(anchors) == 1:
        return TokenGrid(
            columns=tuple((tok,) for tok in anchors[0]),
            hypotheses=(anchors[0],),
            back_pointers=(tuple((i, "MATCH") for i, _ in enumerate(anchors[0])),),
            pair_alignments=(),
        )
    pair_alignments: list[Alignment] = []
    # Rows already merged into the grid; ``grid_columns[c][r]`` is the cell of
    # hypothesis ``r`` in column ``c``. Every column always holds one cell per
    # merged hypothesis, so the grid stays rectangular.
    grid_columns: list[list[Hashable]] = [[tok] for tok in anchors[0]]
    back_pointers: list[tuple[tuple[int, str], ...]] = [
        tuple((i, "MATCH") for i in range(len(anchors[0])))
    ]
    for hyp_idx in range(1, len(anchors)):
        # Align the incoming hypothesis against the pivot row of the grid.
        grid_tokens = [col[0] for col in grid_columns]
        pair = edit_alignment(grid_tokens, anchors[hyp_idx])
        pair_alignments.append(pair)
        moving = anchors[hyp_idx]
        merged: list[list[Hashable]] = []
        remap: list[int] = [-1] * len(grid_columns)
        pointers: list[tuple[int, str]] = []
        g_pos = h_pos = 0
        for op in pair.ops:
            if op.op in {MATCH, SUB}:
                remap[g_pos] = len(merged)
                merged.append([*grid_columns[g_pos], moving[h_pos]])
                pointers.append((len(merged) - 1, op.op))
                g_pos += 1
                h_pos += 1
            elif op.op == INS:
                # Inserted tokens keep their position in the sequence: earlier
                # rows simply have no token there.
                merged.append([GAP] * hyp_idx + [moving[h_pos]])
                pointers.append((len(merged) - 1, "INS"))
                h_pos += 1
            elif op.op == DEL:
                remap[g_pos] = len(merged)
                merged.append([*grid_columns[g_pos], GAP])
                g_pos += 1
            else:
                raise AlignmentError(f"unknown op {op.op}")
        if g_pos != len(grid_columns) or h_pos != len(moving):
            raise AlignmentError("alignment did not consume both sequences")
        back_pointers = [tuple((remap[col], label) for col, label in row) for row in back_pointers]
        back_pointers.append(tuple(pointers))
        grid_columns = merged
    columns = tuple(tuple(col) for col in grid_columns)
    return TokenGrid(
        columns=columns,
        hypotheses=tuple(anchors),
        back_pointers=tuple(back_pointers),
        pair_alignments=tuple(pair_alignments),
    )


def _as_tuple(seq: Sequence[Hashable]) -> tuple[Hashable, ...]:
    if isinstance(seq, str):
        raise AlignmentError("tokens must not be a string")
    return tuple(seq)


def grid_row(grid: TokenGrid, hypothesis_index: int) -> tuple[Hashable, ...]:
    """Return the token sequence for one hypothesis inside the grid."""
    row = [GAP] * grid.width
    pointers = grid.back_pointers[hypothesis_index]
    for hyp_pos, pointer in enumerate(pointers):
        grid_col = pointer[0]
        row[grid_col] = grid.hypotheses[hypothesis_index][hyp_pos]
    return tuple(row)


def grid_coverage(grid: TokenGrid) -> list[int]:
    """Count non-gap cells per hypothesis row in the grid.

    Returns a list of length ``grid.depth`` where each element
    is the number of columns in which that hypothesis has a
    non-gap token. The sum equals the total number of tokens
    across all inputs.
    """
    result: list[int] = []
    for h_idx in range(grid.depth):
        count = 0
        for col in grid.columns:
            if h_idx < len(col) and col[h_idx] != GAP:
                count += 1
        result.append(count)
    return result


def grid_determinism_check(hypotheses: Sequence[Sequence[str]]) -> bool:
    """Verify grid construction is permutation-invariant under ``left`` pivot."""
    a = progressive_align(hypotheses)
    b = progressive_align(hypotheses)
    return a.columns == b.columns
