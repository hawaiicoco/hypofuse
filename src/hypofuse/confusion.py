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
    """Pick the pivot from each slot (minimum-cost path over the slot chain).

    Because the confusion network is a chain with no cross-slot
    constraints, the minimum-cost path (cost = -log posterior) decomposes
    into independent per-slot argmax choices. The per-slot pivot is
    therefore identical to the global minimal-cut 1-best. Provided as an
    explicit function so downstream tooling has a stable call.
    """
    return network.one_best()


_CONFUSION_SCHEMA = "hypofuse.confusion"
_CONFUSION_SCHEMA_VERSION = 1


def confusion_to_json(network: ConfusionNetwork) -> str:
    """Serialize a confusion network as versioned JSON.

    Output is deterministic: ``sort_keys=True`` and the default separators
    ensure byte-stable roundtrips.
    """
    import json

    obj = {
        "schema": _CONFUSION_SCHEMA,
        "schema_version": _CONFUSION_SCHEMA_VERSION,
        "slots": network.to_dict(),
    }
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def confusion_from_json(payload: str) -> ConfusionNetwork:
    """Parse the versioned JSON form of a confusion network.

    Raises ``ValueError`` for missing fields, unknown schema, or
    unsupported schema version.
    """
    import json

    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise ValueError("expected a JSON object with schema and slots")
    for key in ("schema", "schema_version", "slots"):
        if key not in raw:
            raise ValueError(f"missing required field: {key!r}")
    if raw["schema"] != _CONFUSION_SCHEMA:
        raise ValueError(f"unknown schema: {raw['schema']!r}")
    if raw["schema_version"] != _CONFUSION_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {raw['schema_version']!r}")
    slots: list[ConfusionSlot] = []
    for item in raw["slots"]:
        pivot = str(item["pivot"])
        arcs = tuple(Arc(token=a["token"], posterior=float(a["posterior"])) for a in item["arcs"])
        slots.append(ConfusionSlot(pivot=pivot, arcs=arcs))
    return ConfusionNetwork(slots=tuple(slots))


def rover_diff(
    network: ConfusionNetwork,
    rover_tokens: Sequence[str],
) -> list[int]:
    """Return positions where one-best diverges from rover tokens.

    Compares token-by-token. Positions beyond the shorter sequence are
    reported as mismatches. Null/gap tokens are compared literally:
    ``"*"`` equals ``"*"`` and nothing else.
    """
    best = list(network.one_best())
    rover = list(rover_tokens)
    mismatches: list[int] = []
    for i in range(max(len(best), len(rover))):
        if i >= len(best) or i >= len(rover) or best[i] != rover[i]:
            mismatches.append(i)
    return mismatches


def consistent_with_rover(
    network: ConfusionNetwork,
    rover_tokens: Sequence[str],
    pivot_tie_break: str = "lexicographic",
) -> bool:
    """Check that the confusion network's 1-best matches ROVER under matching rules."""
    return len(rover_diff(network, rover_tokens)) == 0


def confusion_to_manifest_row(
    network: ConfusionNetwork,
    utterance_id: str,
    system: str = "fusion",
) -> dict[str, object]:
    """Produce a ``hypofuse.fusion_run`` manifest row from a confusion network.

    The row is compatible with :func:`manifests.validate.validate_record`.
    """
    from hypofuse.manifests import SCHEMA_FUSION_RUN, SCHEMA_VERSION

    tokens = list(network.one_best())
    confidences: list[float] = []
    arcs_list: list[dict[str, object]] = []
    for slot in network.slots:
        pivot_arc = next(
            (a for a in slot.arcs if str(a.token) == slot.pivot),
            slot.arcs[0],
        )
        confidences.append(pivot_arc.posterior)
        arcs_list.append(
            {
                "pivot": slot.pivot,
                "candidates": [[str(a.token), a.posterior] for a in slot.arcs],
            }
        )
    return {
        "schema": SCHEMA_FUSION_RUN,
        "schema_version": SCHEMA_VERSION,
        "utterance_id": utterance_id,
        "systems": [system],
        "tokens": tokens,
        "confidences": confidences,
        "policy": "confusion",
        "config_hash": "",
        "arcs": arcs_list,
    }


def confusion_from_manifest_row(row: dict[str, object]) -> ConfusionNetwork:
    """Reconstruct a confusion network from a ``hypofuse.fusion_run`` row."""
    arcs_raw = row.get("arcs", [])
    slots: list[ConfusionSlot] = []
    for entry in arcs_raw:
        pivot = str(entry["pivot"])
        arcs = tuple(Arc(token=c[0], posterior=float(c[1])) for c in entry["candidates"])
        slots.append(ConfusionSlot(pivot=pivot, arcs=arcs))
    return ConfusionNetwork(slots=tuple(slots))
