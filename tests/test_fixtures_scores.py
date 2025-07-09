"""Realistic scores ordered by edit distance."""

from __future__ import annotations

import pytest

from hypofuse.alignment import edit_alignment
from hypofuse.fixtures import FixtureConfig, generate_fixture


def test_score_ordering_matches_error_ordering() -> None:
    cfg = FixtureConfig(
        n_utterances=50,
        n_best=4,
        seed=42,
        substitution_rate=0.1,
        insertion_rate=0.05,
        deletion_rate=0.05,
    )
    fx = generate_fixture(cfg)
    for u in fx:
        errors = [edit_alignment(u.reference, hyp).errors for hyp in u.hypotheses]
        scores = list(u.acoustic_log10s)
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                if errors[i] < errors[j]:
                    assert scores[i] > scores[j], (
                        f"errors {errors[i]}<{errors[j]} but "
                        f"scores {scores[i]:.4f}<={scores[j]:.4f}"
                    )
                elif errors[i] > errors[j]:
                    assert scores[i] < scores[j]
                else:
                    assert scores[i] == pytest.approx(scores[j])
