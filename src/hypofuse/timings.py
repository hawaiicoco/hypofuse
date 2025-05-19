"""Token timing handling for ASR post-processing.

Provides dataclasses and utilities for working with per-token timing
information, alignment of timings across reference and hypothesis tracks,
and various timing transformations. Pure Python (no numpy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenTiming:
    """A single token with start and end time in seconds.

    Frozen to keep timing records immutable once constructed.
    """

    token: str
    start_s: float
    end_s: float

    def validate(self) -> None:
        """Raise :class:`ValueError` if the timing is invalid.

        Rejects NaN values (checked with :func:`math.isnan`), negative
        values, and intervals where ``end_s < start_s``.
        """
        if math.isnan(self.start_s) or math.isnan(self.end_s):
            raise ValueError("timing values must not be NaN")
        if self.start_s < 0.0 or self.end_s < 0.0:
            raise ValueError("timing values must not be negative")
        if self.end_s < self.start_s:
            raise ValueError("end_s must not precede start_s")

    @property
    def duration_s(self) -> float:
        """Duration of this token in seconds."""
        return self.end_s - self.start_s
