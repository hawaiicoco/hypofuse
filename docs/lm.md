# Language Model: Counting, Smoothing, and ARPA

## Counting and vocabulary

`NgramLM.train(sentences, order=3)` counts all n-grams from 1 to
`order` in the padded sentences. Padding uses `(order-1)` copies
of `<s>` before each sentence and one `</s>` after.

The vocabulary always includes `<unk>`, `<s>`, `</s>`, plus every
token seen in training. OOV tokens at query time are mapped to `<unk>`.

## Katz backoff

`prob_katz(ngram)` implements a simplified Katz backoff:

1. If the n-gram has a non-zero count, return `count / prefix_count`.
2. Otherwise, compute the leftover probability mass `beta` and
   distribute it proportionally to the lower-order distribution.

**Simplifications** compared to the full Katz (1987) estimator:
- No count cutoff threshold (all observed counts are used directly).
- No Turing-Good discounting for low counts.
- The backoff weight `beta` is computed as `(total - observed_mass) / total`
  which is a first-order approximation.

These simplifications keep the implementation minimal and transparent.
The model is suitable for small-vocabulary rescoring tasks and
synthetic experiments but not for large-scale language modeling.

## Jelinek-Mercer interpolation

`prob_jelinek(ngram, lambdas=None)` computes:

```
P(w | h) = sum over k in [1..n] of lambda_k * P_ML(w | h[-k+1:])
```

where `P_ML` is the maximum-likelihood estimate from counts. Default
lambdas are uniform: `1/order` for each order. Custom lambdas must
sum to 1.0 (within tolerance 1e-6).

No EM-based lambda fitting is provided; the user supplies lambdas
directly.

## Perplexity

```
PPL(tokens) = exp( -1/N * sum_i log P(w_i | history) )
```

where `N` is the number of n-gram evaluations in the padded sequence.
Returns 1.0 for empty input. `method="katz"` or `method="jelinek"`
selects the probability estimator.

### OOV policy

Unseen tokens are mapped to `<unk>`. If `<unk>` has zero count in
the training data, the unigram probability falls back to
`1 / total_unigrams`.

## ARPA export/import

### Layout

```
\data\
ngram 1=<count>
ngram 2=<count>
...

\1-grams:
<log10prob>\t<token>
...

\2-grams:
<log10prob>\t<token1> <token2>\t<backoff_weight>
...

\end\
```

Log probabilities are base-10. Backoff weights are included for
n-grams of order 2 and above.

### Losslessness

`to_arpa` followed by `from_arpa` preserves:
- The vocabulary (all tokens in all n-grams).
- The n-gram order.
- The relative ordering of n-grams within each section (sorted).

However, the roundtrip stores raw log10 probabilities as counts
(from `from_arpa`), not the original count table. Probability queries
on the reloaded model use these stored values directly.

## Rescoring

`rescore_lm(hyp, lm)` computes:

```
lm_log10 = sum_i log10 P(w_i | history)
```

using Katz backoff probabilities over the padded token sequence.

`shallow_fusion_score(hyp, lm, lm_weight, acoustic_weight=1.0)`:

```
score = acoustic_weight * acoustic_log10 + lm_weight * lm_log10
```

`rescore_nbest` re-ranks hypotheses by this combined score descending.

There is no explicit word insertion penalty beyond the n-gram
probabilities; the LM score already accounts for sequence probability.

## Probability dispatch: `prob(ngram, method="auto")`

`NgramLM.prob` dispatches to the named method:

| Method | Behaviour |
|---|---|
| `"auto"` | ARPA tables if `source="arpa"`, else Katz |
| `"katz"` | Katz backoff from counts |
| `"jelinek"` | Jelinek-Mercer interpolation |
| `"arpa"` | ARPA log-prob tables with backoff |

## ARPA backoff weight rule

When observed continuations consume all mass,
backoff weight is 1.0 (log10 = 0.0).

## Unigram fallback

Unlisted unigrams fall back to `<unk>` probability,
then to `1 / len(vocab)`.

## Export precision

ARPA uses `%.4f` formatting. Roundtrip equality
is bounded by this quantization.
