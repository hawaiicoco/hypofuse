"""Manifest rows carry the acoustic and LM scores the fixtures generated.

Without them the score-weighted fusion policy and n-best rescoring would see
only the default 0.0 for every hypothesis, so the emitted rows have to match
``FixtureUtterance.acoustic_log10s`` / ``lm_log10s`` rank by rank.
"""

from __future__ import annotations

import math

from hypofuse.fixtures import (
    FixtureConfig,
    FixtureUtterance,
    as_manifest_dicts,
    generate_fixture,
)


def _nbest_rows() -> list[dict]:
    fixtures = generate_fixture(FixtureConfig(n_utterances=3, n_best=3, seed=11))
    rows = as_manifest_dicts(fixtures)
    return [r for r in rows if r["schema"] == "hypofuse.nbest"], fixtures


def test_scores_match_the_generated_fixture() -> None:
    rows, fixtures = _nbest_rows()
    for row, fx in zip(rows, fixtures, strict=True):
        for hyp, acoustic, lm in zip(
            row["hypotheses"], fx.acoustic_log10s, fx.lm_log10s, strict=True
        ):
            assert hyp["acoustic_log10"] == acoustic
            assert hyp["lm_log10"] == lm


def test_scores_are_finite_and_negative_log_likelihoods() -> None:
    rows, _ = _nbest_rows()
    for row in rows:
        for hyp in row["hypotheses"]:
            assert math.isfinite(hyp["acoustic_log10"])
            assert math.isfinite(hyp["lm_log10"])
            # log10 likelihoods of a synthetic decoder are negative.
            assert hyp["acoustic_log10"] < 0.0
            assert hyp["lm_log10"] < 0.0


def test_missing_scores_fall_back_to_zero() -> None:
    """A fixture without score tuples still produces valid rows."""
    bare = FixtureUtterance(
        utterance_id="u0",
        reference=("a", "b"),
        hypotheses=(("a", "b"),),
        speaker_group="A",
        intent_domain="qa",
        noise_db=10.0,
        duration_s=1.0,
    )
    rows = as_manifest_dicts([bare])
    hyp = rows[0]["hypotheses"][0]
    assert hyp["acoustic_log10"] == 0.0
    assert hyp["lm_log10"] == 0.0
