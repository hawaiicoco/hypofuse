# Fusion: ROVER Voting and Confusion Networks

## Progressive alignment grid

`progressive_align` builds a rectangular token grid by performing
pairwise edit alignments between successive hypotheses. The first
hypothesis anchors the grid columns; each subsequent hypothesis is
aligned against the current grid's pivot row, and new columns are
inserted where the incoming hypothesis has tokens not covered by
existing columns. Gap positions are filled with `GAP = "*"`.

The result is a `TokenGrid` where:
- `columns[c][h]` is the token hypothesis `h` contributes at column `c`
  (or `*` if it has no token there).
- `width` = number of columns, `depth` = number of hypotheses.

## Voting policies

### Majority (`"majority"`)

Each candidate token receives weight 1.0. The token with the highest
total weight wins. Equivalent to simple plurality voting.

### Score-weighted (`"score_weighted"`)

Each candidate receives the acoustic score from its hypothesis as
its weight (or 1.0 if no scores are provided). The token with the
highest total weighted vote wins.

### LM-weighted (`"lm_weighted"`)

Each candidate's weight is `score + alpha * lm_score`. This combines
acoustic and language model evidence before voting.

## Tie-break rules

- `"lexicographic"` (default): among tied tokens, pick the one that
  sorts first lexicographically.
- `"first"`: among tied tokens, pick the one that appeared first
  in the candidate list.

## Null/epsilon policies

Columns where every hypothesis has a gap token produce a null output
token (`null_token`, default `"*"`). Gap tokens are excluded from
voting unless the column is entirely gaps.

## Fused confidence

For each column, the fused confidence is:

```
confidence = (number of candidates agreeing with chosen token) / (total candidates)
```

This is the simple agreement fraction, not a calibrated probability.

## Confusion network construction

`build_confusion_network` converts a `TokenGrid` into a chain of
`ConfusionSlot` objects. Each slot corresponds to one grid column.

### Arc posterior estimation

Each hypothesis contributes equal weight (or a caller-supplied weight)
to its token in each column. The column vote totals are normalized to
sum to 1.0 across non-gap tokens (by default) or across all tokens
including epsilon (when `keep_epsilon=True`).

```
posterior(token) = sum_of_weights_for_token / total_weight_in_column
```

### Pivot selection

The pivot is the non-gap arc with the highest posterior. Ties are
broken lexicographically. If only gap arcs exist, the gap becomes
the pivot.

### Arc ordering

Arcs within each slot are sorted by posterior descending, then by
token string ascending for deterministic output.

## 1-best extraction

`one_best()` returns the pivot of each slot. `minimal_cut_one_best()`
returns the same result because the confusion network is a linear
chain with no cross-slot constraints -- the per-slot argmax is the
global minimum-cost path.

## Consistency with ROVER

`consistent_with_rover(network, rover_tokens)` checks that the
confusion network 1-best matches the tokens produced by `fuse()`
under the same policy. `rover_diff` returns the list of positions
where the two diverge.

The confusion network and ROVER agree when both use majority voting
with the same tie-break rule, because both select the highest-vote
token per column.

## Worked example

Three hypotheses aligned on a 3-column grid:

| Column | Hyp A | Hyp B | Hyp C |
|---|---|---|---|
| 0 | the | the | the |
| 1 | cat | bat | cat |
| 2 | sat | sat | * |

Majority vote:
- Column 0: the(3) -> "the", confidence 1.0
- Column 1: cat(2) > bat(1) -> "cat", confidence 0.667
- Column 2: sat(2) -> "sat", confidence 1.0

Confusion network posteriors:
- Slot 0: the=1.0
- Slot 1: cat=0.667, bat=0.333
- Slot 2: sat=1.0
