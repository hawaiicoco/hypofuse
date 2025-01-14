"""Smoke tests for NBestList / NBestHypothesis dataclasses."""

from __future__ import annotations

from hypofuse.manifests.nbest import NBestHypothesis, NBestList


def test_nbest_hypothesis_defaults() -> None:
    h = NBestHypothesis(rank=1, text="hello")
    assert h.tokens == ()
    assert h.posteriors == ()
    assert h.acoustic_log10 == 0.0


def test_nbest_list_returns_ranks() -> None:
    nbest = NBestList(
        utterance_id="utt_1",
        system="sys_a",
        language="en",
        hypotheses=(
            NBestHypothesis(rank=1, text="hello"),
            NBestHypothesis(rank=2, text="hello world"),
        ),
    )
    assert nbest.ranks() == (1, 2)


def test_nbest_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    import pytest

    h = NBestHypothesis(rank=1, text="x")
    with pytest.raises(FrozenInstanceError):
        h.rank = 2  # type: ignore[misc]
