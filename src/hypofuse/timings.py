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


@dataclass(frozen=True)
class TimingTrack:
    """An ordered sequence of non-overlapping token timings.

    ``tolerance_s`` controls the floating-point slack allowed when deciding
    whether two adjacent tokens overlap. The default ``1e-9`` is tight enough
    to catch real overlaps while absorbing rounding noise.
    """

    tokens: tuple[TokenTiming, ...] = ()
    tolerance_s: float = 1e-9

    def validate(self) -> None:
        """Validate every token and reject overlaps.

        Touching boundaries (``end_s == next start_s``) are allowed.
        Non-finite values (NaN, inf) are rejected with :class:`ValueError`.
        """
        for i, tt in enumerate(self.tokens):
            tt.validate()
            if not math.isfinite(tt.start_s) or not math.isfinite(tt.end_s):
                raise ValueError(f"non-finite timing value at index {i}")
            if i > 0:
                prev_end = self.tokens[i - 1].end_s
                if tt.start_s < prev_end - self.tolerance_s:
                    raise ValueError(
                        f"overlapping tokens at index {i}: "
                        f"start_s={tt.start_s} < previous end_s={prev_end}"
                    )
