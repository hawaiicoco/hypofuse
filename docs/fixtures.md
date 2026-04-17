# Synthetic Data Factory

The `hypofuse.fixtures` module generates ASR-like data with **known**
ground truth. All output is **synthetic** -- it is not real ASR output,
not derived from any speech corpus, and should not be cited as
benchmark results.

## FixtureConfig fields

| Field | Default | Effect |
|---|---|---|
| `n_utterances` | 20 | Number of synthetic utterances |
| `n_best` | 3 | Hypotheses per utterance |
| `substitution_rate` | 0.05 | Per-token substitution probability |
| `insertion_rate` | 0.02 | Per-token insertion probability |
| `deletion_rate` | 0.02 | Per-token deletion probability |
| `speaker_groups` | `("A","B","C")` | Group labels (random assignment) |
| `intents` | `("greeting","qa","command")` | Intent domain labels |
| `noise_db_range` | `(-10.0, 35.0)` | SNR range in dB |
| `duration_range` | `(0.5, 12.0)` | Duration range in seconds |
| `seed` | 0 | Random seed for determinism |
| `confusables` | `()` | Pairs `(correct, confused)` for substitutions |
| `group_bias` | `{}` | Per-group error rate multiplier |
| `noise_sensitivity` | 0.0 | How much noise scales error rates |
| `rank_decay` | 0.0 | Extra error budget for later n-best entries |

Validation: rate sum must not exceed 1.0; group bias keys must be
in `speaker_groups`; `noise_sensitivity` in [-10, 10]; `rank_decay >= 0`.

## Error model

For each reference token, the generator draws a uniform random number
and applies:
1. Substitution (probability `sub_rate`): replace with a random token
   from `confusables` or `["x","y","z"]`.
2. Insertion (probability `ins_rate`): keep the token and append a
   random extra token.
3. Deletion (probability `del_rate`): skip the token.
4. Otherwise: keep the token unchanged.

### Effective rates

The base rates are scaled by three factors:

- **Group bias**: `rate *= group_bias[group]` when the group is in
  the bias map.
- **Noise sensitivity**: `factor = 1 + noise_sensitivity * (1 - noise_norm)`
  where `noise_norm = (noise_db - lo) / (hi - lo)`. Lower SNR
  increases error rates when `noise_sensitivity > 0`.
- **Rank decay**: `factor = 1 + rank_decay * rank`. Later n-best
  entries get more errors.

All rates are clamped to [0, 1] individually.

## Score model

Acoustic and LM log10 scores are derived from the error count:

```
acoustic_log10 = -10.0 - 2.0 * errors - 0.01 * (35 - noise_db)
lm_log10       = -5.0 - 1.5 * errors
```

Fewer errors yields higher (less negative) scores. This is a
synthetic scoring model, not a real acoustic/language model.

## Known ground truth in tests

Tests use `generate_fixture` with specific seeds and configurations
to assert known error rates. For example:
- `substitution_rate=0.0` with `insertion_rate=0.0` and
  `deletion_rate=0.0` produces perfect hypotheses (WER = 0).
- `FixtureConfig(n_utterances=2000, seed=42, substitution_rate=0.05, ...)`
  produces measured rates within 0.01 of the configured rates.

## Explicit statement

All fixture output is synthetic. No real ASR system, pretrained model,
or speech corpus is involved. The generator exists to provide
deterministic test data with known ground truth so that metrics,
fusion, and calibration code can be tested against fixed targets.

## Worked recipe: assert a known error rate

```python
from hypofuse.fixtures import FixtureConfig, generate_fixture
from hypofuse.alignment import word_error_rate

cfg = FixtureConfig(
    n_utterances=500,
    n_best=1,
    seed=42,
    substitution_rate=0.10,
    insertion_rate=0.0,
    deletion_rate=0.0,
)
fixtures = generate_fixture(cfg)
rates = []
for fx in fixtures:
    ref = list(fx.reference)
    hyp = list(fx.hypotheses[0])
    rates.append(word_error_rate(ref, hyp))
mean_wer = sum(rates) / len(rates)
# With 500 utterances and sub_rate=0.10, the measured WER
# should be close to 0.10 (within ~0.02).
assert abs(mean_wer - 0.10) < 0.03
```
