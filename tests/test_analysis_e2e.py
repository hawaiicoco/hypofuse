"""Deterministic end-to-end golden test (drift detector).

Uses a tiny synthetic fixture so the golden stays readable.  All data
is synthetic; no real ASR benchmark is implied.
"""

from __future__ import annotations

from hypofuse.analysis import (
    UtteranceScore,
    paired_bootstrap_ci,
    report_to_markdown,
    slice_by_duration,
    slice_metrics,
)
from hypofuse.fixtures import FixtureConfig, generate_fixture

GOLDEN = "# Error Analysis Report\n\n## Slices\n| Slice | Count | CER | WER |\n|---|---|---|---|\n| all | 5 | 0.0000 | 0.0583 |\n| &lt;1s | 1 | 0.0000 | 0.1250 |\n| &gt;10s | 2 | 0.0000 | 0.0000 |\n| &lt;10s | 2 | 0.0000 | 0.0833 |\n\n## System comparisons (paired bootstrap, 95% CI)\n| Pair | n | Δ CER | CER CI | Δ WER | WER CI |\n|---|---|---|---|---|---|\n| A vs B | 5 | -0.0247 | [-0.1435, 0.1098] | -0.0250 | [-0.1167, 0.0842] |"


def test_end_to_end_golden() -> None:
    cfg = FixtureConfig(n_utterances=5, n_best=2, seed=42)
    futs = generate_fixture(cfg)

    scores_a: list[UtteranceScore] = []
    scores_b: list[UtteranceScore] = []
    for f in futs:
        hyp_a = f.hypotheses[0]
        hyp_b = f.hypotheses[1]
        scores_a.append(
            UtteranceScore(
                f.utterance_id,
                f.reference,
                hyp_a,
                f.speaker_group,
                f.noise_db,
                f.duration_s,
                f.intent_domain,
            )
        )
        scores_b.append(
            UtteranceScore(
                f.utterance_id,
                f.reference,
                hyp_b,
                f.speaker_group,
                f.noise_db,
                f.duration_s,
                f.intent_domain,
            )
        )

    slices = slice_by_duration(scores_a)
    metrics = slice_metrics(slices, use_cer=False)
    comp = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=100, seed=42)
    md = report_to_markdown(metrics, [comp])
    assert md == GOLDEN
