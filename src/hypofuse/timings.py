"""Token timing handling for ASR post-processing.

Provides dataclasses and utilities for working with per-token timing
information, alignment of timings across reference and hypothesis tracks,
and various timing transformations. Pure Python (no numpy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from hypofuse.alignment import DEL, INS, MATCH, SUB, edit_alignment
from hypofuse.exceptions import AlignmentError, SchemaError


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

    @property
    def total_s(self) -> float:
        """Total span from first start to last end.

        Returns ``0.0`` for empty tracks.
        """
        if not self.tokens:
            return 0.0
        return self.tokens[-1].end_s - self.tokens[0].start_s

    def speaking_ratio(self) -> float:
        """Fraction of :attr:`total_s` occupied by speaking tokens.

        Computed as ``sum(duration_s) / total_s``. Returns ``0.0`` for
        empty tracks or tracks with zero span.
        """
        span = self.total_s
        if span == 0.0:
            return 0.0
        speaking = sum(t.duration_s for t in self.tokens)
        return speaking / span

    def __len__(self) -> int:
        return len(self.tokens)


def from_manifest_row(row: Any) -> TimingTrack:
    """Parse an n-best manifest row into a :class:`TimingTrack`.

    Expected shape::

        {"tokens": ["a", "b"], "timings": [{"start_s": 0.0, "end_s": 0.5}, ...]}

    Raises :class:`~hypofuse.exceptions.SchemaError` when the row is
    malformed (missing fields, wrong types, length mismatch, or invalid
    timing values).
    """
    if not isinstance(row, dict):
        raise SchemaError("row must be a dict")
    if "tokens" not in row:
        raise SchemaError("missing 'tokens' field")
    if "timings" not in row:
        raise SchemaError("missing 'timings' field")
    tokens_field = row["tokens"]
    timings_field = row["timings"]
    if not isinstance(tokens_field, list):
        raise SchemaError("'tokens' must be a list")
    if not isinstance(timings_field, list):
        raise SchemaError("'timings' must be a list")
    if len(tokens_field) != len(timings_field):
        raise SchemaError("'tokens' and 'timings' must have the same length")
    built: list[TokenTiming] = []
    for idx, (tok, tmg) in enumerate(zip(tokens_field, timings_field, strict=True)):
        if not isinstance(tok, str):
            raise SchemaError(f"token at index {idx} must be a string")
        if not isinstance(tmg, dict):
            raise SchemaError(f"timing at index {idx} must be a dict")
        if "start_s" not in tmg or "end_s" not in tmg:
            raise SchemaError(f"timing at index {idx} missing start_s/end_s")
        start = tmg["start_s"]
        end = tmg["end_s"]
        if not isinstance(start, (int, float)) or isinstance(start, bool):
            raise SchemaError(f"start_s at index {idx} must be numeric")
        if not isinstance(end, (int, float)) or isinstance(end, bool):
            raise SchemaError(f"end_s at index {idx} must be numeric")
        built.append(TokenTiming(token=tok, start_s=float(start), end_s=float(end)))
    track = TimingTrack(tokens=tuple(built))
    track.validate()
    return track


def to_manifest_row(track: TimingTrack) -> dict[str, Any]:
    """Serialize a :class:`TimingTrack` to the n-best manifest dict shape.

    The returned dict has keys ``"tokens"`` (list of str) and ``"timings"``
    (list of ``{"start_s": ..., "end_s": ...}`` dicts).
    """
    return {
        "tokens": [tt.token for tt in track.tokens],
        "timings": [{"start_s": tt.start_s, "end_s": tt.end_s} for tt in track.tokens],
    }


@dataclass(frozen=True)
class TimingAlignmentStep:
    """One step in a timing-aware alignment between reference and hypothesis.

    ``op`` is one of ``"match"``, ``"substitution"``, ``"insertion"``,
    ``"deletion"`` (lowercase to distinguish from the alignment module
    constants). ``ref_index`` and ``hyp_index`` are ``None`` when the
    corresponding side has no token for this step.
    """

    op: str
    ref_index: int | None
    hyp_index: int | None
    start_s: float
    end_s: float


def align_timings(
    reference_track: TimingTrack,
    hypothesis_track: TimingTrack,
) -> tuple[TimingAlignmentStep, ...]:
    """Map hypothesis token timings onto reference positions.

    Runs :func:`hypofuse.alignment.edit_alignment` on the token strings and
    attaches timing information from whichever side owns the token:

    * ``match`` / ``substitution`` -- both sides present; start/end come
      from the hypothesis track.
    * ``insertion`` -- hypothesis only; start/end from hypothesis.
    * ``deletion`` -- reference only; start/end from reference.
    """
    ref_tokens = [tt.token for tt in reference_track.tokens]
    hyp_tokens = [tt.token for tt in hypothesis_track.tokens]
    alignment = edit_alignment(ref_tokens, hyp_tokens)

    steps: list[TimingAlignmentStep] = []
    ref_idx = 0
    hyp_idx = 0
    for op in alignment.ops:
        if op.op == MATCH:
            ht = hypothesis_track.tokens[hyp_idx]
            steps.append(
                TimingAlignmentStep(
                    op="match",
                    ref_index=ref_idx,
                    hyp_index=hyp_idx,
                    start_s=ht.start_s,
                    end_s=ht.end_s,
                )
            )
            ref_idx += 1
            hyp_idx += 1
        elif op.op == SUB:
            ht = hypothesis_track.tokens[hyp_idx]
            steps.append(
                TimingAlignmentStep(
                    op="substitution",
                    ref_index=ref_idx,
                    hyp_index=hyp_idx,
                    start_s=ht.start_s,
                    end_s=ht.end_s,
                )
            )
            ref_idx += 1
            hyp_idx += 1
        elif op.op == INS:
            ht = hypothesis_track.tokens[hyp_idx]
            steps.append(
                TimingAlignmentStep(
                    op="insertion",
                    ref_index=None,
                    hyp_index=hyp_idx,
                    start_s=ht.start_s,
                    end_s=ht.end_s,
                )
            )
            hyp_idx += 1
        elif op.op == DEL:
            rt = reference_track.tokens[ref_idx]
            steps.append(
                TimingAlignmentStep(
                    op="deletion",
                    ref_index=ref_idx,
                    hyp_index=None,
                    start_s=rt.start_s,
                    end_s=rt.end_s,
                )
            )
            ref_idx += 1
        else:
            raise AlignmentError(f"unexpected alignment op: {op.op}")
    return tuple(steps)


def interpolate_gaps(track: TimingTrack, max_gap_s: float) -> TimingTrack:
    """Fill zero-duration tokens by splitting the surrounding silence evenly.

    For each token whose ``duration_s`` is zero, the silence before it
    (``start_s - prev_end``) and after it (``next_start - end_s``) is each
    halved and the token is expanded into the middle of that window.

    ``max_gap_s`` caps the *total* surrounding gap (``next_start - prev_end``)
    eligible for interpolation; tokens whose gap exceeds this threshold are
    left unchanged. Raises :class:`ValueError` when ``max_gap_s`` is negative.

    The transformation preserves the track span (first ``start_s`` and last
    ``end_s`` are unchanged) and the token order.
    """
    if max_gap_s < 0.0:
        raise ValueError("max_gap_s must not be negative")
    if not track.tokens:
        return track

    tokens = list(track.tokens)
    result: list[TokenTiming] = []
    for i, tt in enumerate(tokens):
        if tt.duration_s > 0.0:
            result.append(tt)
            continue
        prev_end = tokens[i - 1].end_s if i > 0 else tt.start_s
        next_start = tokens[i + 1].start_s if i < len(tokens) - 1 else tt.end_s
        total_gap = next_start - prev_end
        if total_gap <= 0.0 or total_gap > max_gap_s:
            result.append(tt)
            continue
        new_start = (prev_end + tt.start_s) / 2.0
        new_end = (tt.end_s + next_start) / 2.0
        result.append(TokenTiming(token=tt.token, start_s=new_start, end_s=new_end))
    return TimingTrack(tokens=tuple(result), tolerance_s=track.tolerance_s)


def _round_half_away_from_zero(x: float) -> int:
    """Round *x* to the nearest integer, breaking ties away from zero.

    This is the classical "round half up" rule for positive values and
    "round half down" for negative values, matching the behaviour expected
    by most audio frame-grid quantizers.
    """
    if x >= 0.0:
        return math.floor(x + 0.5)
    return math.ceil(x - 0.5)


def snap_to_grid(track: TimingTrack, frame_s: float = 0.01) -> TimingTrack:
    """Quantize every timing boundary to a multiple of ``frame_s``.

    Uses round-half-away-from-zero (documented in
    :func:`_round_half_away_from_zero`) so the result is deterministic and
    independent of the platform's banker's-rounding default.

    The snapped track never has negative boundaries or inverted intervals:
    ``start_s`` is clamped to ``>= 0`` and ``end_s >= start_s`` is enforced
    after snapping.
    """
    if frame_s <= 0.0:
        raise ValueError("frame_s must be positive")
    if not track.tokens:
        return track

    result: list[TokenTiming] = []
    for tt in track.tokens:
        snapped_start = _round_half_away_from_zero(tt.start_s / frame_s) * frame_s
        snapped_end = _round_half_away_from_zero(tt.end_s / frame_s) * frame_s
        snapped_start = max(0.0, snapped_start)
        snapped_end = max(0.0, snapped_end)
        if snapped_end < snapped_start:
            snapped_end = snapped_start
        result.append(TokenTiming(token=tt.token, start_s=snapped_start, end_s=snapped_end))
    return TimingTrack(tokens=tuple(result), tolerance_s=track.tolerance_s)


def duration_buckets(
    total_s: float,
    edges: tuple[float, ...] = (1.0, 3.0, 8.0),
) -> str:
    """Classify a duration into a human-readable bucket label.

    Boundary values land in the *upper* bucket: a duration of exactly
    ``1.0`` is labelled ``"1.0-3.0s"``, not ``"<1.0s"``. The lowest bucket
    captures everything strictly below ``edges[0]``; the highest bucket
    captures everything at or above ``edges[-1]``.

    Default edges produce labels: ``"<1.0s"``, ``"1.0-3.0s"``,
    ``"3.0-8.0s"``, ``">=8.0s"``.
    """
    for i, edge in enumerate(edges):
        if total_s < edge:
            if i == 0:
                return f"<{edge}s"
            return f"{edges[i - 1]}-{edge}s"
    return f">={edges[-1]}s"


@dataclass(frozen=True)
class TimingReport:
    """Summary statistics for a collection of timing tracks.

    ``mean_s``, ``median_s``, and ``p95_s`` describe the distribution of
    per-track total durations. ``gap_total_s`` sums every inter-token
    silence across all tracks.
    """

    count: int
    mean_s: float
    median_s: float
    p95_s: float
    gap_total_s: float


def _nearest_rank_quantile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank quantile (deterministic, no interpolation).

    The rank is ``ceil(q * n)``, clamped to ``[1, n]``. This rule picks
    the smallest value whose cumulative position is at or above the
    requested percentile. Returns ``0.0`` for an empty input.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    rank = math.ceil(q * n)
    rank = max(1, min(rank, n))
    return sorted_values[rank - 1]


def timing_report(tracks: list[TimingTrack]) -> TimingReport:
    """Compute summary statistics for a list of timing tracks.

    * ``count`` -- number of tracks.
    * ``mean_s`` -- arithmetic mean of per-track :attr:`TimingTrack.total_s`.
    * ``median_s`` / ``p95_s`` -- nearest-rank quantiles (see
      :func:`_nearest_rank_quantile`).
    * ``gap_total_s`` -- sum of every inter-token silence across all tracks.

    Pure Python implementation; does not depend on scipy or numpy.
    """
    if not tracks:
        return TimingReport(count=0, mean_s=0.0, median_s=0.0, p95_s=0.0, gap_total_s=0.0)

    durations: list[float] = []
    gap_total = 0.0
    for track in tracks:
        durations.append(track.total_s)
        for j in range(1, len(track.tokens)):
            gap = track.tokens[j].start_s - track.tokens[j - 1].end_s
            gap_total += max(0.0, gap)

    count = len(durations)
    mean_s = sum(durations) / count
    sorted_d = sorted(durations)
    median_s = _nearest_rank_quantile(sorted_d, 0.5)
    p95_s = _nearest_rank_quantile(sorted_d, 0.95)

    return TimingReport(
        count=count,
        mean_s=mean_s,
        median_s=median_s,
        p95_s=p95_s,
        gap_total_s=gap_total,
    )
