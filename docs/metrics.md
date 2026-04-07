# Error Metrics: CER, WER, and Averaging

## Definitions

Both CER and WER are computed via `edit_alignment`, a standard
Levenshtein-style dynamic-programming alignment over token sequences.

### Edit distance recurrence

For reference `r[0..n-1]` and hypothesis `h[0..m-1]`:

```
dp[0][0] = 0
dp[i][0] = dp[i-1][0] + c_del      (deletion)
dp[0][j] = dp[0][j-1] + c_ins      (insertion)
dp[i][j] = min(
    dp[i-1][j-1] + (0 if r[i-1]==h[j-1] else c_sub),   # match/sub
    dp[i][j-1] + c_ins,                                  # insertion
    dp[i-1][j] + c_del                                   # deletion
)
```

Default costs: `c_sub = c_ins = c_del = 1.0` (unit-cost Levenshtein).

### Tie-break order

When multiple operations achieve the same minimum cost, the tie is
broken in the order: substitution, insertion, deletion. This makes
the alignment deterministic.

### WER

```
WER = (S + I + D) / N
```

where `S` = substitutions, `I` = insertions, `D` = deletions,
`N` = reference length (number of reference tokens).

### CER

Identical formula applied at the character level: the reference and
hypothesis strings are split into individual characters before alignment.

### Empty-reference policy

| Reference | Hypothesis | Rate |
|---|---|---|
| empty | empty | 0.0 |
| empty | non-empty | 1.0 |
| non-empty | (any) | errors / ref_length |

### Symmetric error rate

```
symmetric = errors / mean(ref_length, hyp_length)
```

Range is `[0, 2]`. Returns 0.0 when both are empty.

## Micro vs macro averaging

- **Micro** (`corpus_error_rate(average="micro")`): total errors across
  all utterances divided by total reference tokens. Gives more weight
  to longer utterances.
- **Macro** (`corpus_error_rate(average="macro")`): arithmetic mean of
  per-utterance error rates. Each utterance contributes equally
  regardless of length.

These differ when utterances have different reference lengths.

## Relation to jiwer

The `jiwer` library computes WER using the same Levenshtein distance
and the same denominator (reference length). hypofuse's `word_error_rate`
and `character_error_rate` produce the same values as jiwer's `wer()`
and `cer()` for the same token inputs, given the same cost model
(unit costs). hypofuse additionally supports custom substitution
costs via `AlignmentCosts.token_costs`, banded alignment via the
`band` parameter, and explicit micro/macro averaging over corpora.

## Hand-computed toy examples

### Example 1: one substitution

```
ref = ["the", "cat", "sat"]
hyp = ["the", "bat", "sat"]
```

Alignment: MATCH(the), SUB(cat->bat), MATCH(sat).
S=1, I=0, D=0, N=3. WER = 1/3 = 0.3333.

### Example 2: one insertion

```
ref = ["cat", "sat"]
hyp = ["cat", "on", "sat"]
```

Alignment: MATCH(cat), INS(on), MATCH(sat).
S=0, I=1, D=0, N=2. WER = 1/2 = 0.5.

### Example 3: one deletion

```
ref = ["the", "big", "cat"]
hyp = ["the", "cat"]
```

Alignment: MATCH(the), DEL(big), MATCH(cat).
S=0, I=0, D=1, N=3. WER = 1/3 = 0.3333.

### Example 4: CER for Chinese

```
ref = "你好世界"  (4 characters: 你 好 世 界)
hyp = "你好时间"  (4 characters: 你 好 时 间)
```

Character alignment: MATCH(你), MATCH(好), SUB(世->时), SUB(界->间).
S=2, I=0, D=0, N=4. CER = 2/4 = 0.5.

### Example 5: micro vs macro

Two utterances:
- Utterance A: ref_length=2, errors=1 -> rate=0.5
- Utterance B: ref_length=8, errors=2 -> rate=0.25

Micro: (1+2)/(2+8) = 3/10 = 0.3
Macro: (0.5+0.25)/2 = 0.375
