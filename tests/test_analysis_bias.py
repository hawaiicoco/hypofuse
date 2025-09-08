"""Group bias table: per-group error rate with confidence intervals."""

from __future__ import annotations

from hypofuse.analysis import UtteranceScore, group_bias_table


def test_bad_group_sorted_first() -> None:
    scores = [
        UtteranceScore("u0", ("a", "b"), ("a", "b"), speaker_group="good"),
        UtteranceScore("u1", ("a", "b"), ("a", "b"), speaker_group="good"),
        UtteranceScore("u2", ("a", "b"), ("x", "y"), speaker_group="bad"),
        UtteranceScore("u3", ("a", "b"), ("x", "y"), speaker_group="bad"),
    ]
    rows = group_bias_table(scores, "speaker_group", n_bootstrap=100, seed=0)
    assert rows[0].group == "bad"
    assert rows[0].wer > rows[-1].wer


def test_counts_sum_to_total() -> None:
    scores = [
        UtteranceScore(f"u{i}", ("a",), ("a",), speaker_group="A" if i % 2 == 0 else "B")
        for i in range(10)
    ]
    rows = group_bias_table(scores, "speaker_group", n_bootstrap=50, seed=0)
    total = sum(r.count for r in rows)
    assert total == len(scores)


def test_ci_brackets_point_estimate() -> None:
    scores = [
        UtteranceScore(
            f"u{i}",
            ("a", "b"),
            ("a", "b") if i % 3 else ("x", "b"),
            speaker_group="G",
        )
        for i in range(20)
    ]
    rows = group_bias_table(scores, "speaker_group", n_bootstrap=200, seed=0)
    for r in rows:
        assert r.wer_ci_low <= r.wer <= r.wer_ci_high
