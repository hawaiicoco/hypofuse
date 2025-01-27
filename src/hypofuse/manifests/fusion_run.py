"""Fusion run records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionArc:
    """A single alignment position with arc posteriors per hypothesis."""

    pivot: str
    candidates: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class FusionRun:
    """The result of fusing N hypothesis lists for one utterance.

    `systems` are the system ids that contributed hypotheses; `tokens`
    are the fused output tokens; `confidences` are fused per-token
    confidences aligned with `tokens` (same length).
    """

    utterance_id: str
    systems: tuple[str, ...]
    tokens: tuple[str, ...]
    confidences: tuple[float, ...]
    policy: str
    config_hash: str = ""
    arcs: tuple[FusionArc, ...] = ()

    def num_tokens(self) -> int:
        return len(self.tokens)
