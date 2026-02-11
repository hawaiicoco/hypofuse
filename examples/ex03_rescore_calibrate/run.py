"""Example ex03: rescore with n-gram LM and calibrate confidences.

Fully offline, deterministic and synthetic.  No network access, no real
ASR output, no benchmark claims.  The language model is trained on a
tiny in-script corpus and the calibration numbers come from synthetic
data with a constructed relationship, not from a real ASR system.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from hypofuse.analysis import (
    UtteranceScore,
    report_to_markdown,
    slice_by_field,
    slice_metrics,
)
from hypofuse.confidence import (
    expected_calibration_error,
    reliability_bins,
    temperature_scale,
)
from hypofuse.fixtures import FixtureConfig, generate_fixture
from hypofuse.ngram import NgramLM
from hypofuse.rescore import ScoredHypothesis, lm_weight_sweep


def main(outdir: str) -> int:
    """Run the ex03 example, writing a report into *outdir*."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)

    # 1. Train a tiny n-gram LM on an in-script synthetic corpus.
    corpus: list[list[str]] = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "ran", "in", "the", "park"],
        ["we", "saw", "the", "cat", "in", "the", "park"],
        ["hello", "world", "good", "morning"],
        ["goodbye", "evening", "good", "night"],
    ]
    lm = NgramLM.train(corpus, order=2)
    print(f"trained {lm.order}-gram LM, vocab={len(lm.vocab)}, unigrams={lm.total_unigrams}")

    # 2. Build n-best list and run LM-weight sweep.
    cfg = FixtureConfig(n_utterances=5, n_best=3, seed=99)
    fixtures = generate_fixture(cfg)
    fx = fixtures[0]
    nbest = [
        ScoredHypothesis.from_tokens(h, acoustic_log10=fx.acoustic_log10s[i])
        for i, h in enumerate(fx.hypotheses)
    ]
    ref = list(fx.reference)
    weights: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 1.0)
    sweep = lm_weight_sweep(nbest, ref, lm, weights=weights)
    print(f"{'weight':>8} {'metric':>8} chosen")
    print("-" * 44)
    for w, metric, text in sweep:
        print(f"{w:>8.2f} {metric:>8.4f} {text}")

    # 3. Calibrate per-token confidences with temperature scaling.
    raw_scores: list[float] = [0.9, 0.8, 0.7, 0.3, 0.1]
    calibrated = temperature_scale(raw_scores, temperature=1.5)
    cal_str = ", ".join(f"{c:.4f}" for c in calibrated)
    print(f"calibrated: [{cal_str}]")

    # 4. ECE / reliability bins with synthetic labels.
    # Constructed relationship: higher confidence -> more likely correct.
    confidences: list[float] = [0.9, 0.8, 0.7, 0.4, 0.2, 0.6, 0.5, 0.3, 0.1, 0.95]
    accuracies: list[float] = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ece = expected_calibration_error(confidences, accuracies, n_bins=5)
    bins = reliability_bins(confidences, accuracies, n_bins=5)
    print(f"ECE = {ece:.4f} (synthetic labels, constructed relationship)")
    for b in bins:
        print(
            f"  [{b.lower:.1f}, {b.upper:.1f}] "
            f"n={b.count} conf={b.avg_confidence:.3f} acc={b.avg_accuracy:.3f}"
        )

    # 5. Write Markdown error-analysis report.
    items: list[UtteranceScore] = [
        UtteranceScore(
            utterance_id=utt.utterance_id,
            reference=utt.reference,
            hypothesis=utt.hypotheses[0],
            speaker_group=utt.speaker_group,
            intent_domain=utt.intent_domain,
            noise_db=utt.noise_db,
            duration_s=utt.duration_s,
        )
        for utt in fixtures
    ]
    slices = slice_by_field(items, "speaker_group")
    metrics = slice_metrics(slices)
    meta: dict[str, str] = {
        "note": "constructed data, not real ASR",
        "source": "synthetic",
    }
    md = report_to_markdown(metrics, [], meta=meta)
    (target / "report.md").write_text(md, encoding="utf-8")
    print("wrote error-analysis report")

    return 0


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="ex03_")
    raise SystemExit(main(outdir))
