"""Hand-computed rescore_lm verification for a tiny LM.

The arithmetic is shown inline so reviewers can check every term.
"""

from __future__ import annotations

import math

import pytest

from hypofuse.ngram import NgramLM
from hypofuse.rescore import ScoredHypothesis, rescore_lm


def test_rescore_lm_equals_hand_computed_sum() -> None:
    """rescore_lm must equal the sum of per-ngram log10 probabilities.

    Training corpus: [["a", "b"]], order 2.
    Padded:  [<s>, a, b, </s>]
    Unigrams: <s>:1  a:1  b:1  </s>:1   total_unigrams = 4
    Bigrams:  (<s>,a):1  (a,b):1  (b,</s>):1

    rescore_lm start-pads with EOS, so ["a", "b"] is scored as
    [</s>, a, b, </s>] over the bigrams (</s>,a), (a,b), (b,</s>).

    prob_katz((</s>, a)) -- unseen bigram, so it backs off:
        prefix (</s>,) has total = 1 and no observed continuations,
        hence beta = 1 and alpha = beta / total = 1;
        prefix_prob = P(</s>) = 1/4  ->  1 * 0.25 = 0.25
    prob_katz((a, b)) = count(a,b) / count(a) = 1 / 1 = 1.0
    prob_katz((b, </s>)) = count(b,</s>) / count(b) = 1 / 1 = 1.0

    score = log10(0.25) + log10(1.0) + log10(1.0) = log10(0.25)
    """
    lm = NgramLM.train([["a", "b"]], order=2)
    hyp = ScoredHypothesis.from_tokens(["a", "b"])
    expected = math.log10(0.25) + math.log10(1.0) + math.log10(1.0)
    assert rescore_lm(hyp, lm) == pytest.approx(expected)
    assert lm.prob_katz(("</s>", "a")) == pytest.approx(0.25)
