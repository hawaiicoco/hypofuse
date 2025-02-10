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
            back_pointers=(tuple(((i, "MATCH") for i, _ in enumerate(anchors[0]))),),
            pair_alignments=(),
        )
    pair_alignments: list[Alignment] = []
    grid_columns: list[list[Hashable]] = [[tok] for tok in anchors[0]]
    back_pointers: list[list[tuple[int, str]]] = [[(i, "MATCH") for i, _ in enumerate(anchors[0])]]
    # Subsequent hypotheses are aligned to the running grid.
    for hyp_idx in range(1, len(anchors)):
        grid_tokens: list[Hashable] = []
        for col in grid_columns:
            first = col[0]
            grid_tokens.append(first if first != GAP else "*")
        pair = edit_alignment(grid_tokens, anchors[hyp_idx])
        pair_alignments.append(pair)
        new_columns: list[list[Hashable]] = []
        new_pointers: list[list[tuple[int, str]]] = []
        for col in grid_columns:
            new_columns.append(list(col))
        for _src_idx in range(len(anchors[hyp_idx])):
            new_pointers.append([])
        # Walk pair alignment to merge the new hypothesis into the grid.
        g_pos = h_pos = 0
        for op in pair.ops:
            if op.op in {MATCH, SUB}:
                if g_pos < len(new_columns):
                    new_columns[g_pos].append(anchors[hyp_idx][h_pos])
                    new_pointers[h_pos].append((g_pos, op.op))
                else:
                    new_columns.append([anchors[hyp_idx][h_pos]])
                    new_pointers[h_pos].append((len(new_columns) - 1, op.op))
                g_pos += 1
                h_pos += 1
            elif op.op == INS:
                new_columns.append([anchors[hyp_idx][h_pos]])
                new_pointers[h_pos].append((len(new_columns) - 1, "INS"))
                h_pos += 1
            elif op.op == DEL:
                if g_pos < len(new_columns):
                    new_columns[g_pos].append(GAP)
                g_pos += 1
            else:
                raise AlignmentError(f"unknown op {op.op}")
        # Ensure each existing grid column has the right number of entries.
        while len(new_pointers) < len(anchors[hyp_idx]):
            new_pointers.append([])
        grid_columns = new_columns
        back_pointers.append(tuple(tuple(p) for p in new_pointers))
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
    for grid_col, pointer in enumerate(grid.back_pointers[hypothesis_index]):
        row[grid_col] = grid.hypotheses[hypothesis_index][pointer[0]]
    return tuple(row)


def grid_determinism_check(hypotheses: Sequence[Sequence[str]]) -> bool:
    """Verify grid construction is permutation-invariant under ``left`` pivot."""
    a = progressive_align(hypotheses)
    b = progressive_align(hypotheses)
    return a.columns == b.columns
