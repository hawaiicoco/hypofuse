"""Optional neural confidence model on synthetic data.

This module demonstrates the package's confidence features end to end
using a small MLP trained on synthetic posteriors. No pretrained weights
are shipped or downloaded, and no real ASR benchmark is implied. The
neural model serves as an integration exercise; downstream users should
evaluate on their own labelled data before drawing conclusions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from hypofuse.util import seeded

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE: bool = True
except ImportError:
    TORCH_AVAILABLE = False

FEATURE_NAMES: tuple[str, ...] = (
    "max_posterior",
    "mean_posterior",
    "entropy",
    "posterior_margin",
    "agreement_ratio",
    "n_tokens",
    "acoustic_log10",
    "lm_log10",
)


@dataclass(frozen=True)
class ConfidenceFeatures:
    """Fixed-width float vector summarising a posterior distribution.

    ``entropy`` uses natural log (nats). ``posterior_margin`` is
    ``top1 - top2`` of the sorted posterior. ``agreement_ratio`` falls
    back to ``max_posterior`` when no vote counts are supplied.
    """

    max_posterior: float
    mean_posterior: float
    entropy: float
    posterior_margin: float
    agreement_ratio: float
    n_tokens: int
    acoustic_log10: float
    lm_log10: float

    def as_list(self) -> list[float]:
        """Return the feature vector as a plain list matching FEATURE_NAMES."""
        return [
            self.max_posterior,
            self.mean_posterior,
            self.entropy,
            self.posterior_margin,
            self.agreement_ratio,
            float(self.n_tokens),
            self.acoustic_log10,
            self.lm_log10,
        ]


def confidence_features(
    posteriors: Sequence[float],
    vote_counts: Sequence[int] | None = None,
    acoustic_log10: float | None = None,
    lm_log10: float | None = None,
) -> ConfidenceFeatures:
    """Extract a fixed-width feature vector from a posterior distribution.

    ``posteriors`` is the probability mass for each candidate token at a
    single position. Entropy uses natural log (nats). Raises ``ValueError``
    on empty input or mismatched optional lengths.
    """
    if not posteriors:
        raise ValueError("posteriors must not be empty")
    if vote_counts is not None and len(vote_counts) != len(posteriors):
        raise ValueError("vote_counts length must match posteriors")

    n = len(posteriors)
    max_p = max(posteriors)
    mean_p = sum(posteriors) / n

    entropy = 0.0
    for p in posteriors:
        if p > 0.0:
            entropy -= p * math.log(p)

    sorted_p = sorted(posteriors, reverse=True)
    top2 = sorted_p[1] if n > 1 else 0.0
    margin = sorted_p[0] - top2

    if vote_counts is not None:
        total_votes = sum(vote_counts)
        agreement = max(vote_counts) / total_votes if total_votes > 0 else 0.0
    else:
        agreement = max_p

    acoustic = acoustic_log10 if acoustic_log10 is not None else 0.0
    lm = lm_log10 if lm_log10 is not None else 0.0

    return ConfidenceFeatures(
        max_posterior=max_p,
        mean_posterior=mean_p,
        entropy=entropy,
        posterior_margin=margin,
        agreement_ratio=agreement,
        n_tokens=n,
        acoustic_log10=acoustic,
        lm_log10=lm,
    )


def feature_matrix(rows: Sequence[ConfidenceFeatures]) -> list[list[float]]:
    """Convert a sequence of feature objects to a list-of-lists matrix."""
    return [r.as_list() for r in rows]


def synthetic_confidence_dataset(
    n: int = 400,
    seed: int = 0,
    error_rate: float = 0.2,
) -> tuple[list[list[float]], list[int]]:
    """Generate synthetic labelled confidence data.

    Correct tokens (label 1) draw the emitted candidate's posterior ``p``
    from Beta(6, 2) (mean 0.75); incorrect tokens (label 0) draw it from
    Beta(2, 6) (mean 0.25). The remaining mass is split over two rivals,
    ``[p, (1 - p) / 2, (1 - p) / 2]``, so a correct token is peaked on the
    emitted candidate (max posterior about 0.75) while an incorrect one is
    peaked on a rival (max posterior about 0.375). This is a controlled
    synthetic generator with a constructed relationship; it does not model
    real ASR output.
    """
    rng = seeded(seed)
    features: list[list[float]] = []
    labels: list[int] = []
    for _ in range(n):
        is_correct = rng.random() >= error_rate
        label = 1 if is_correct else 0
        p = rng.betavariate(6, 2) if is_correct else rng.betavariate(2, 6)
        rival = (1.0 - p) / 2.0
        feat = confidence_features([p, rival, rival])
        features.append(feat.as_list())
        labels.append(label)
    return features, labels


def baseline_accuracy(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Accuracy using score >= 0.5 as a positive prediction.

    Returns 0.0 on empty input.
    """
    if not scores:
        return 0.0
    if len(scores) != len(labels):
        raise ValueError("scores and labels must align")
    correct = sum(1 for s, lbl in zip(scores, labels, strict=True) if (s >= 0.5) == (lbl == 1))
    return correct / len(scores)


def auc_from_scores(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Pure-Python AUC via the Mann-Whitney U statistic.

    Ties are resolved by assigning average rank (the standard
    convention). Returns 0.5 when all scores are tied and 1.0 for
    perfect separation. Raises ``ValueError`` when either class is
    empty or lengths mismatch.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must align")
    n_pos = sum(1 for lbl in labels if lbl == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both classes must be present")

    indexed = sorted(enumerate(scores), key=lambda x: x[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j

    rank_sum_pos = sum(r for r, lbl in zip(ranks, labels, strict=True) if lbl == 1)
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u_stat / (n_pos * n_neg)


def logistic_temperature_scale(
    scores: Sequence[float],
    temperature: float = 1.0,
) -> list[float]:
    """Apply logistic temperature scaling to each score independently.

    Each score is treated as a probability, converted to a logit,
    divided by ``temperature``, then mapped back through the sigmoid.
    Temperature > 1 flattens toward 0.5; temperature < 1 sharpens.
    Raises ``ValueError`` when temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    out: list[float] = []
    for s in scores:
        s_clamped = max(1e-7, min(1.0 - 1e-7, s))
        logit = math.log(s_clamped / (1.0 - s_clamped))
        scaled_logit = logit / temperature
        out.append(1.0 / (1.0 + math.exp(-scaled_logit)))
    return out


def fit_temperature(
    scores: Sequence[float],
    labels: Sequence[int],
    grid: Sequence[float] | None = None,
) -> float:
    """Fit a temperature parameter by grid search minimising ECE.

    Torch-free: uses :func:`logistic_temperature_scale` and
    :func:`hypofuse.confidence.expected_calibration_error`. The default
    grid includes 1.0 (identity), so the result is always at least as
    good as no scaling.
    """
    from hypofuse.confidence import expected_calibration_error

    if grid is None:
        grid = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 10.0]
    if len(scores) != len(labels):
        raise ValueError("scores and labels must align")
    accuracies = [float(lbl) for lbl in labels]
    best_t = 1.0
    best_ece = float("inf")
    for t in grid:
        scaled = logistic_temperature_scale(list(scores), temperature=t)
        ece = expected_calibration_error(scaled, accuracies)
        if ece < best_ece:
            best_ece = ece
            best_t = t
    return best_t


class NeuralConfidenceModel:
    """Tiny MLP: input -> hidden (tanh) -> 1 (logistic sigmoid).

    Training is deterministic: both ``torch.manual_seed`` and Python
    ``random.seed`` are set from ``seed`` at construction and again at
    the start of :meth:`fit`.
    """

    def __init__(self, input_dim: int, hidden: int = 8, seed: int = 0) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "the 'torch' extra is required for NeuralConfidenceModel; "
                "install with: pip install hypofuse[torch]"
            )
        import torch

        self._input_dim = input_dim
        self._hidden = hidden
        self._seed = seed
        torch.manual_seed(seed)
        self._fc1 = torch.nn.Linear(input_dim, hidden)
        self._fc2 = torch.nn.Linear(hidden, 1)

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
        epochs: int = 30,
        lr: float = 0.05,
    ) -> list[float]:
        """Train the model; return per-epoch mean binary cross-entropy loss."""
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "the 'torch' extra is required for NeuralConfidenceModel; "
                "install with: pip install hypofuse[torch]"
            )
        import random

        import torch

        torch.manual_seed(self._seed)
        random.seed(self._seed)
        X = torch.tensor([list(row) for row in features], dtype=torch.float32)
        y = torch.tensor(list(labels), dtype=torch.float32).unsqueeze(1)
        params = list(self._fc1.parameters()) + list(self._fc2.parameters())
        optimizer = torch.optim.SGD(params, lr=lr)
        losses: list[float] = []
        for _ in range(epochs):
            optimizer.zero_grad()
            h = torch.tanh(self._fc1(X))
            logits = self._fc2(h)
            probs = torch.sigmoid(logits)
            clamped = probs.clamp(1e-7, 1.0 - 1e-7)
            loss = torch.nn.functional.binary_cross_entropy(clamped, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        return losses

    def predict_proba(self, features: Sequence[Sequence[float]]) -> list[float]:
        """Return P(label=1) for each row, in [0, 1]."""
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "the 'torch' extra is required for NeuralConfidenceModel; "
                "install with: pip install hypofuse[torch]"
            )
        import torch

        with torch.no_grad():
            X = torch.tensor([list(row) for row in features], dtype=torch.float32)
            h = torch.tanh(self._fc1(X))
            logits = self._fc2(h)
            probs = torch.sigmoid(logits)
        return [float(p) for p in probs.squeeze(1).tolist()]

    def state_dict_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the shape of each named parameter tensor."""
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "the 'torch' extra is required for NeuralConfidenceModel; "
                "install with: pip install hypofuse[torch]"
            )
        out: dict[str, tuple[int, ...]] = {}
        for name, param in list(self._fc1.named_parameters(prefix="fc1")) + list(
            self._fc2.named_parameters(prefix="fc2")
        ):
            out[name] = tuple(param.shape)
        return out


def calibrate_with_model(
    model: NeuralConfidenceModel,
    features: Sequence[Sequence[float]],
) -> list[float]:
    """Return calibrated probabilities from a trained neural model."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "the 'torch' extra is required for calibrate_with_model; "
            "install with: pip install hypofuse[torch]"
        )
    return model.predict_proba(features)
