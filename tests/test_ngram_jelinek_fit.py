"""Tests for NgramLM.fit_jelinek_lambdas (deleted interpolation by EM)."""

from __future__ import annotations

import math

import pytest

from hypofuse.ngram import BOS, EOS, NgramLM


def _train_and_held_out() -> tuple[NgramLM, list[list[str]]]:
    train = [
        ["a", "b", "c"],
        ["b", "c", "a"],
        ["c", "a", "b"],
        ["a", "b", "c"],
    ]
    held_out = [["a", "b", "c"], ["c", "a", "b"]]
    lm = NgramLM.train(train, order=2)
    return lm, held_out


def test_lambdas_sum_to_one() -> None:
    lm, held_out = _train_and_held_out()
    lambdas = lm.fit_jelinek_lambdas(held_out, iterations=5)
    assert sum(lambdas) == pytest.approx(1.0)


def test_lambdas_all_non_negative() -> None:
    lm, held_out = _train_and_held_out()
    lambdas = lm.fit_jelinek_lambdas(held_out, iterations=5)
    assert all(lam >= 0 for lam in lambdas)


def test_lambdas_concentrate_on_a_deterministic_corpus() -> None:
    """Every context here has exactly one continuation.

    Deleted interpolation should therefore move almost all mass to the
    highest order; the lambdas still sum to one, stay inside [0, 1], and are
    reproducible for a fixed input.
    """
    sents = [
        ["a", "b", "c", "d"],
        ["b", "c", "d", "a"],
        ["c", "d", "a", "b"],
        ["d", "a", "b", "c"],
    ]
    lm = NgramLM.train(sents, order=2)
    lambdas = lm.fit_jelinek_lambdas(sents, iterations=5)
    assert sum(lambdas) == pytest.approx(1.0)
    assert all(0.0 <= lam <= 1.0 for lam in lambdas)
    assert lambdas[-1] > 0.9
    assert lm.fit_jelinek_lambdas(sents, iterations=5) == lambdas


def test_empty_held_out_returns_uniform() -> None:
    lm = NgramLM.train([["a", "b"]], order=2)
    lambdas = lm.fit_jelinek_lambdas([], iterations=5)
    assert lambdas == pytest.approx((0.5, 0.5))


def _order_probabilities(lm: NgramLM, padded: list[str], i: int) -> list[float]:
    """Maximum-likelihood estimate of the token at *i* for each order 1..n."""
    probs: list[float] = []
    for k in range(1, lm.order + 1):
        start = i - k + 1
        if start < 0:
            probs.append(0.0)
            continue
        ngram = tuple(lm.map_sequence(padded[start : i + 1]))
        if k == 1:
            probs.append(lm.counts[1].get(ngram, 0) / max(lm.total_unigrams, 1))
        else:
            context = ngram[:-1]
            count_ngram = lm.counts[k].get(ngram, 0)
            count_context = lm.counts[k - 1].get(context, 0)
            probs.append(count_ngram / count_context if count_context else 0.0)
    return probs


def _held_out_log_likelihood(
    lm: NgramLM, held_out: list[list[str]], lambdas: tuple[float, ...]
) -> float:
    """The objective EM maximises, written out from its definition."""
    total = 0.0
    for sent in held_out:
        padded = [BOS] * (lm.order - 1) + sent + [EOS]
        for i in range(lm.order - 1, len(padded)):
            probs = _order_probabilities(lm, padded, i)
            mixture = sum(lam * p for lam, p in zip(lambdas, probs, strict=True))
            total += math.log(max(mixture, 1e-12))
    return total


def test_fitted_lambdas_improve_held_out_log_likelihood() -> None:
    """EM cannot lower the held-out mixture log-likelihood it maximises.

    Perplexity is deliberately *not* compared: the components are maximum
    likelihood estimates, so an unseen held-out n-gram contributes a floor
    term and the ranking can flip without the fit being wrong. Synthetic
    corpus, no benchmark claim.
    """
    lm, held_out = _train_and_held_out()
    uniform = (0.5, 0.5)
    fitted = lm.fit_jelinek_lambdas(held_out, iterations=10)
    assert _held_out_log_likelihood(lm, held_out, fitted) >= (
        _held_out_log_likelihood(lm, held_out, uniform) - 1e-9
    )


def _perplexity_with_lambdas(lm: NgramLM, tokens: list[str], lambdas: tuple[float, ...]) -> float:
    """Compute perplexity using specific Jelinek lambdas."""
    mapped = lm.map_sequence(tokens)
    padded = [BOS] * (lm.order - 1) + mapped + [EOS]
    log_sum = 0.0
    n = 0
    for i in range(lm.order - 1, len(padded)):
        ng = tuple(padded[i - lm.order + 1 : i + 1])
        p = lm.prob_jelinek(ng, lambdas=lambdas)
        log_sum -= math.log(max(p, 1e-12))
        n += 1
    return math.exp(log_sum / max(1, n))


def test_perplexity_with_fitted_lambdas_is_finite() -> None:
    lm, held_out = _train_and_held_out()
    fitted = lm.fit_jelinek_lambdas(held_out, iterations=10)
    tokens = [t for s in held_out for t in s]
    perplexity = _perplexity_with_lambdas(lm, tokens, fitted)
    assert math.isfinite(perplexity)
    assert perplexity > 0.0
