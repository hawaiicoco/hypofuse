"""Token and utterance confidence computations."""

from __future__ import annotations

import pytest

from hypofuse.confidence import token_confidence_from_posteriors, utterance_confidence_from_vote


def test_token_confidence_mean() -> None:
    assert token_confidence_from_posteriors([0.4, 0.6, 0.8]) == pytest.approx(0.6)


def test_token_confidence_empty() -> None:
    assert token_confidence_from_posteriors([]) == 0.0


def test_utterance_confidence_agreement() -> None:
    votes = [["a", "b"], ["a", "x"], ["a", "b"]]
    confs = utterance_confidence_from_vote(votes, ["a", "b"])
    assert confs[0] == pytest.approx(1.0)
    assert confs[1] == pytest.approx(2 / 3)


def test_utterance_confidence_handles_short_streams() -> None:
    votes = [["a", "b"], ["a"]]
    confs = utterance_confidence_from_vote(votes, ["a", "b"])
    assert confs[0] == pytest.approx(1.0)
    # Only one stream covers col 1 and it agrees; confidence is 1.0 over the
    # observed (non-missing) votes.
    assert confs[1] == pytest.approx(1.0)


def test_utterance_confidence_empty_input() -> None:
    assert utterance_confidence_from_vote([], []) == []
