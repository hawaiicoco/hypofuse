"""Confusion networks built from a multi-hypothesis alignment grid.

A confusion network is a sequence of slots; each slot has an arc set where
one arc is the "pivot" (the most likely token) and the rest are alternatives.
Arcs carry normalized posterior probabilities so they sum to 1 per slot.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from hypofuse.multi_align import GAP, TokenGrid


@dataclass(frozen=True)
class Arc:
    token: Hashable
    posterior: float


@dataclass(frozen=True)
class ConfusionSlot:
    pivot: str
    arcs: tuple[Arc, ...]

    def is_consistent(self) -> bool:
        total = sum(a.posterior for a in self.arcs)
        return abs(total - 1.0) < 1e-6


@dataclass(frozen=True)
class ConfusionNetwork:
    slots: tuple[ConfusionSlot, ...]

    def one_best(self) -> tuple[str, ...]:
        return tuple(slot.pivot for slot in self.slots)

    def to_dict(self) -> list[dict[str, object]]:
        out = []
        for slot in self.slots:
            out.append(
                {
                    "pivot": slot.pivot,
                    "arcs": [{"token": a.token, "posterior": a.posterior} for a in slot.arcs],
                }
            )
        return out

    def validate(self, tol: float = 1e-6) -> None:
        """Check posteriors normalize per slot, non-negative, no empty slots.

        Raises ``ValueError`` naming the offending slot index.
        """
        if not self.slots:
            return
        for idx, slot in enumerate(self.slots):
            if not slot.arcs:
                raise ValueError(f"slot {idx} is empty (no arcs)")
            for arc in slot.arcs:
                if arc.posterior < 0.0:
                    raise ValueError(f"slot {idx}: negative posterior {arc.posterior}")
            total = sum(a.posterior for a in slot.arcs)
            if abs(total - 1.0) > tol:
                raise ValueError(f"slot {idx}: posteriors sum to {total}, expected 1.0 (tol={tol})")


def build_confusion_network(
    grid: TokenGrid,
    weights: Sequence[Sequence[float]] | None = None,
    keep_epsilon: bool = False,
) -> ConfusionNetwork:
    """Build a confusion network from an alignment grid.

    Each column becomes a slot. Per-slot posteriors are normalized to sum
    to 1 across non-gap tokens. The pivot is the token with the highest
    posterior; ties are broken lexicographically.

    When ``keep_epsilon`` is true, gap/epsilon arcs are included in the
    arc set with their computed posterior but are never chosen as pivot
    unless the slot contains no other token. When false (the default),
    gap tokens are excluded from the arc set entirely.
    """
    n_hyps = grid.depth
    score_grid: list[list[float]] = [[1.0] * n_hyps for _ in range(grid.width)]
    if weights is not None:
        for h_idx, row in enumerate(weights):
            for col_idx in range(grid.width):
                if h_idx < n_hyps and col_idx < len(row):
                    score_grid[col_idx][h_idx] = float(row[col_idx])
    slots: list[ConfusionSlot] = []
    for col_idx, column in enumerate(grid.columns):
        counts: dict[Hashable, float] = {}
        for h_idx, token in enumerate(column):
            if token == GAP and not keep_epsilon:
                continue
            w = max(0.0, score_grid[col_idx][h_idx])
            counts[token] = counts.get(token, 0.0) + w
        if not counts:
            slots.append(ConfusionSlot(pivot=GAP, arcs=(Arc(GAP, 1.0),)))
            continue
        total = sum(counts.values())
        arcs: list[Arc] = []
        for token, count in counts.items():
            arcs.append(Arc(token, count / total))
        arcs.sort(key=lambda a: (-a.posterior, str(a.token)))
        non_gap = [a for a in arcs if a.token != GAP]
        pivot = non_gap[0].token if non_gap else arcs[0].token
        slots.append(ConfusionSlot(pivot=str(pivot), arcs=tuple(arcs)))
    return ConfusionNetwork(slots=tuple(slots))


def minimal_cut_one_best(network: ConfusionNetwork) -> tuple[str, ...]:
    """Pick the pivot from each slot.

    With arcs already normalized per slot, the minimum-cut 1-best is just
    the pivot sequence. Provided as an explicit function so downstream
    tooling has a stable call.
    """
    return network.one_best()


def confusion_to_json(network: ConfusionNetwork) -> str:
    """Serialize a confusion network deterministically."""
    import json

    return json.dumps(network.to_dict(), sort_keys=True, ensure_ascii=False)


def confusion_from_json(payload: str) -> ConfusionNetwork:
    """Parse the JSON form of a confusion network."""
    import json

    raw = json.loads(payload)
    slots: list[ConfusionSlot] = []
    for item in raw:
        pivot = str(item["pivot"])
        arcs = tuple(Arc(token=a["token"], posterior=float(a["posterior"])) for a in item["arcs"])
        slots.append(ConfusionSlot(pivot=pivot, arcs=arcs))
    return ConfusionNetwork(slots=tuple(slots))


def consistent_with_rover(
    network: ConfusionNetwork,
    rover_tokens: Sequence[str],
    pivot_tie_break: str = "lexicographic",
) -> bool:
    """Check that the confusion network's 1-best matches ROVER under matching rules."""
    return list(network.one_best()) == list(rover_tokens)
