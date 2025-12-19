"""Token and utterance confidence plus lightweight calibration.

Calibration uses temperature scaling and a deterministic piecewise-linear
relabeller. ECE (Expected Calibration Error) is computed with deterministic
bin boundaries. There is no scipy dependency.

Documentation honest disclaimer: this module calibrates *synthetic* score
distributions on known ground truth. It does not claim state-of-the-art
calibration on real ASR hypotheses; downstream users are expected to
evaluate calibration on their own labeled data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from hypofuse.exceptions import CalibrationError


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    avg_confidence: float
    avg_accuracy: float


def expected_calibration_error(
    confidences: Sequence[float],
    accuracies: Sequence[float],
    n_bins: int = 10,
    binning: str = "uniform",
) -> float:
    """Compute ECE with uniform-width or quantile bins on [0, 1].

    ``binning`` is ``"uniform"`` (default, equal-width bins) or
    ``"quantile"`` (nearest-rank, approximately equal count per bin;
    ties may cause slight count imbalance).
    """
    if len(confidences) != len(accuracies):
        raise ValueError("confidences and accuracies must align")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if binning == "uniform":
        bins = _bin_counts(confidences, accuracies, n_bins)
    elif binning == "quantile":
        bins = _quantile_bin_counts(confidences, accuracies, n_bins)
    else:
        raise ValueError(f"unknown binning: {binning!r}")
    total = max(1, sum(b.count for b in bins))
    ece = sum(abs(b.avg_confidence - b.avg_accuracy) * b.count for b in bins) / total
    return ece


def reliability_bins(
    confidences: Sequence[float],
    accuracies: Sequence[float],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    return _bin_counts(confidences, accuracies, n_bins)


def _bin_counts(
    confidences: Sequence[float], accuracies: Sequence[float], n_bins: int
) -> list[CalibrationBin]:
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for c, a in zip(confidences, accuracies, strict=True):
        idx = min(n_bins - 1, max(0, int(c * n_bins)))
        bins[idx].append((c, a))
    out: list[CalibrationBin] = []
    for i in range(n_bins):
        if not bins[i]:
            out.append(CalibrationBin(i / n_bins, (i + 1) / n_bins, 0, 0.0, 0.0))
            continue
        cs, a_s = zip(*bins[i], strict=True)
        out.append(
            CalibrationBin(
                lower=i / n_bins,
                upper=(i + 1) / n_bins,
                count=len(bins[i]),
                avg_confidence=sum(cs) / len(cs),
                avg_accuracy=sum(a_s) / len(a_s),
            )
        )
    return out


def _quantile_bin_counts(
    confidences: Sequence[float], accuracies: Sequence[float], n_bins: int
) -> list[CalibrationBin]:
    """Bin by quantile: nearest-rank, approximately equal count per bin."""
    n = len(confidences)
    indexed = sorted(range(n), key=lambda i: confidences[i])
    raw: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for rank, idx in enumerate(indexed):
        bin_idx = min(n_bins - 1, rank * n_bins // n) if n > 0 else 0
        raw[bin_idx].append((confidences[idx], accuracies[idx]))
    out: list[CalibrationBin] = []
    for i in range(n_bins):
        if not raw[i]:
            out.append(CalibrationBin(0.0, 0.0, 0, 0.0, 0.0))
            continue
        cs, a_s = zip(*raw[i], strict=True)
        out.append(
            CalibrationBin(
                lower=min(cs),
                upper=max(cs),
                count=len(raw[i]),
                avg_confidence=sum(cs) / len(cs),
                avg_accuracy=sum(a_s) / len(a_s),
            )
        )
    return out


def temperature_scale(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Apply softmax temperature scaling to a list of logits.

    With temperature=1 the distribution is unchanged. Larger values flatten
    the distribution; smaller values sharpen it.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        return []
    import math

    scaled = [s / temperature for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    z = sum(exps)
    return [e / z for e in exps]


def piecewise_calibrate(
    scores: Sequence[float],
    breakpoints: Sequence[float],
    slopes: Sequence[float],
    intercepts: Sequence[float],
) -> list[float]:
    """Apply a deterministic piecewise-linear transform.

    ``breakpoints`` are the lower edges of each segment. The number of
    slopes/intercepts must equal the number of breakpoints. Scores outside
    the outermost breakpoints use the closest segment.
    """
    if len(slopes) != len(intercepts) or len(slopes) != len(breakpoints):
        raise ValueError("slopes, intercepts, and breakpoints must align")
    out: list[float] = []
    for s in scores:
        idx = 0
        for i, bp in enumerate(breakpoints):
            if s >= bp:
                idx = i
        slope = slopes[idx]
        intercept = intercepts[idx]
        out.append(slope * s + intercept)
    return out


def token_confidence_from_posteriors(posteriors: Sequence[float]) -> float:
    """Mean per-token posterior as utterance confidence."""
    if not posteriors:
        return 0.0
    return sum(posteriors) / len(posteriors)


def utterance_confidence_from_vote(
    votes: Sequence[Sequence[str]],
    chosen: Sequence[str],
) -> list[float]:
    """Compute per-token agreement scores from a list of vote streams."""
    if not votes or not chosen:
        return []
    confidences = []
    for col_idx, tok in enumerate(chosen):
        n = 0
        n_match = 0
        for stream in votes:
            if col_idx < len(stream):
                n += 1
                if stream[col_idx] == tok:
                    n_match += 1
        confidences.append(n_match / n if n else 0.0)
    return confidences


def brier_score(probs: Sequence[float], labels: Sequence[float]) -> float:
    """Brier score: mean squared error between probabilities and binary labels.

    Lower is better. Perfect predictions yield 0.0. Raises ``ValueError``
    on mismatched lengths, empty input, or non-binary labels.
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")
    if not probs:
        raise ValueError("probs and labels must not be empty")
    total = 0.0
    for p, y in zip(probs, labels, strict=True):
        if y not in (0, 1):
            raise ValueError("labels must be binary (0 or 1)")
        total += (p - y) ** 2
    return total / len(probs)


def log_loss(probs: Sequence[float], labels: Sequence[float], eps: float = 1e-15) -> float:
    """Log loss (cross-entropy) with clipping to avoid ``log(0)``.

    Probabilities are clipped to ``[eps, 1 - eps]`` before taking the
    logarithm. Lower is better.
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")
    if not probs:
        raise ValueError("probs and labels must not be empty")
    total = 0.0
    for p, y in zip(probs, labels, strict=True):
        if y not in (0, 1):
            raise ValueError("labels must be binary (0 or 1)")
        p_clip = max(eps, min(1.0 - eps, p))
        total += -(y * math.log(p_clip) + (1 - y) * math.log(1.0 - p_clip))
    return total / len(probs)


@dataclass(frozen=True)
class ReliabilityCurve:
    """Reliability diagram data with one entry per bin.

    Empty bins have ``count=0``, ``mean_predicted=0.0`` and
    ``observed_frequency=0.0``. Probabilities exactly on a bin edge
    land in the upper bin, matching :func:`reliability_bins`.
    """

    bin_edges: tuple[float, ...]
    mean_predicted: tuple[float, ...]
    observed_frequency: tuple[float, ...]
    counts: tuple[int, ...]


def reliability_curve(
    probs: Sequence[float],
    labels: Sequence[float],
    bins: int = 10,
) -> ReliabilityCurve:
    """Compute reliability curve data using the same bin rule as reliability_bins."""
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    raw = _bin_counts(probs, labels, bins)
    edges = tuple(i / bins for i in range(bins + 1))
    return ReliabilityCurve(
        bin_edges=edges,
        mean_predicted=tuple(b.avg_confidence for b in raw),
        observed_frequency=tuple(b.avg_accuracy for b in raw),
        counts=tuple(b.count for b in raw),
    )


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True)
class LogisticCalibrator:
    """Platt-style logistic calibrator: ``sigmoid(coef * score + intercept)``.

    Attributes:
        coef: slope of the logistic transform.
        intercept: bias of the logistic transform.
    """

    coef: float
    intercept: float

    def apply(self, scores: Sequence[float]) -> list[float]:
        """Return calibrated probabilities in (0, 1)."""
        return [_sigmoid(self.coef * s + self.intercept) for s in scores]


def fit_logistic(
    scores: Sequence[float],
    labels: Sequence[float],
    iterations: int = 200,
    lr: float = 0.1,
    l2: float = 0.0,
) -> LogisticCalibrator:
    """Fit a logistic calibrator via full-batch gradient descent.

    Minimises ``NLL + (l2 / 2) * coef ** 2`` where
    ``p = sigmoid(coef * s + intercept)``.

    Raises :class:`CalibrationError` when all labels share one class
    (the optimum would diverge).
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise CalibrationError("need at least one sample")
    unique = set(labels)
    if unique <= {0} or unique <= {1}:
        raise CalibrationError("all labels are the same class; logistic fit would diverge")
    coef = 0.0
    intercept = 0.0
    n = len(scores)
    for _ in range(iterations):
        grad_a = 0.0
        grad_b = 0.0
        for s, y in zip(scores, labels, strict=True):
            p = _sigmoid(coef * s + intercept)
            err = p - y
            grad_a += err * s
            grad_b += err
        coef -= lr * (grad_a / n + l2 * coef)
        intercept -= lr * (grad_b / n)
    return LogisticCalibrator(coef=coef, intercept=intercept)


def calibrate(
    scores: Sequence[float],
    labels: Sequence[float],
    method: str = "temperature",
) -> list[float]:
    """Dispatch to a calibration method and return calibrated scores.

    Supported methods: ``"temperature"``, ``"logistic"``, ``"piecewise"``.
    """
    if method == "temperature":
        temp = fit_temperature(scores, labels)
        return [_sigmoid(s / temp) for s in scores]
    if method == "piecewise":
        return piecewise_calibrate(scores, [0.0], [1.0], [0.0])
    if method == "logistic":
        cal = fit_logistic(scores, labels)
        return cal.apply(scores)
    raise ValueError(f"unknown method: {method!r}")


def fit_temperature(
    scores: Sequence[float],
    labels: Sequence[float],
    grid: Sequence[float] | None = None,
    iterations: int | None = None,
) -> float:
    """Find the temperature minimising NLL via deterministic grid search.

    The default grid spans 0.05 to 5.0 in steps of 0.05. ``iterations``
    is accepted for API symmetry but unused (grid search is exhaustive).

    Raises ``ValueError`` for empty or non-positive grid values.
    """
    _ = iterations
    if grid is None:
        grid = [round(0.05 * i, 2) for i in range(1, 101)]
    if not grid:
        raise ValueError("grid must not be empty")
    for t in grid:
        if t <= 0:
            raise ValueError("grid values must be positive")
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("need at least one sample")
    eps = 1e-15

    def _score_nll(temp: float) -> float:
        total = 0.0
        for s, y in zip(scores, labels, strict=True):
            p = max(eps, min(1.0 - eps, _sigmoid(s / temp)))
            total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
        return total

    best_temp = grid[0]
    best_nll = _score_nll(best_temp)
    for t in grid[1:]:
        v = _score_nll(t)
        if v < best_nll:
            best_nll = v
            best_temp = t
    return best_temp
