"""Edge cases for extended fixture config fields."""

from __future__ import annotations

from hypofuse.fixtures import FixtureConfig, generate_fixture


def test_multiple_confusable_pairs() -> None:
    cfg = FixtureConfig(
        n_utterances=20,
        n_best=1,
        seed=42,
        substitution_rate=0.5,
        insertion_rate=0.0,
        deletion_rate=0.0,
        confusables=(("their", "there"), ("to", "too")),
    )
    fx = generate_fixture(cfg)
    all_ref_tokens: set[str] = set()
    for u in fx:
        all_ref_tokens.update(u.reference)
    assert all_ref_tokens.issubset({"their", "to"})


def test_group_bias_partial_groups() -> None:
    cfg = FixtureConfig(
        n_utterances=50,
        n_best=1,
        seed=42,
        substitution_rate=0.1,
        speaker_groups=("A", "B", "C"),
        group_bias={"A": 3.0},
    )
    fx = generate_fixture(cfg)
    groups = {u.speaker_group for u in fx}
    assert groups.issubset({"A", "B", "C"})


def test_rank_decay_with_single_hypothesis() -> None:
    cfg = FixtureConfig(
        n_utterances=10,
        n_best=1,
        seed=42,
        rank_decay=1.0,
    )
    fx = generate_fixture(cfg)
    for u in fx:
        assert len(u.hypotheses) == 1


def test_all_rates_zero_produces_perfect_with_decay() -> None:
    cfg = FixtureConfig(
        n_utterances=5,
        n_best=2,
        seed=42,
        substitution_rate=0.0,
        insertion_rate=0.0,
        deletion_rate=0.0,
        rank_decay=1.0,
    )
    fx = generate_fixture(cfg)
    for u in fx:
        for hyp in u.hypotheses:
            assert hyp == u.reference


def test_lm_scores_independent_of_noise() -> None:
    """LM scores depend only on error count, not noise_db."""
    cfg_clean = FixtureConfig(
        n_utterances=20,
        n_best=2,
        seed=42,
        substitution_rate=0.1,
        noise_db_range=(30.0, 30.0),
    )
    cfg_noisy = FixtureConfig(
        n_utterances=20,
        n_best=2,
        seed=42,
        substitution_rate=0.1,
        noise_db_range=(-10.0, -10.0),
    )
    fx_clean = generate_fixture(cfg_clean)
    fx_noisy = generate_fixture(cfg_noisy)
    for uc, un in zip(fx_clean, fx_noisy, strict=True):
        assert uc.lm_log10s == un.lm_log10s
