"""Noise sensitivity and error count correlation."""

from __future__ import annotations

from hypofuse.alignment import edit_alignment
from hypofuse.fixtures import FixtureConfig, generate_fixture


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation in pure Python."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return cov / (sx * sy)


def test_noise_sensitivity_negative_correlation() -> None:
    """Lower SNR (lower noise_db) should produce more errors.

    With noise_sensitivity=3.0 the factor ranges from 4.0 (noisy)
    to 1.0 (clean), giving a strong negative Pearson r.
    """
    cfg = FixtureConfig(
        n_utterances=400,
        n_best=1,
        seed=42,
        substitution_rate=0.1,
        insertion_rate=0.05,
        deletion_rate=0.05,
        noise_sensitivity=3.0,
    )
    fx = generate_fixture(cfg)
    noise_dbs: list[float] = []
    error_counts: list[float] = []
    for u in fx:
        ops = edit_alignment(u.reference, u.hypotheses[0]).ops
        errors = sum(1 for op in ops if op.op != "MATCH")
        noise_dbs.append(u.noise_db)
        error_counts.append(float(errors))
    r = _pearson(noise_dbs, error_counts)
    assert r < -0.1, f"expected negative correlation, got {r}"
