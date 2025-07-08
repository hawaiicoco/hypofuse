"""Deterministic synthetic fixture factory.

The factory generates ASR-like data with **known** ground truth so every
metric / fusion / calibration test asserts against a fixed target. All
data is synthetic and labelled as such; nothing here is real ASR output.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from hypofuse.alignment import edit_alignment
from hypofuse.util import seeded, stable_hash


@dataclass(frozen=True)
class FixtureConfig:
    """Configuration for synthetic fixture generation.

    All rate fields are per-token competing probabilities; their sum must
    not exceed 1.0.  ``confusables`` replaces placeholder substitution
    tokens when non-empty.  ``group_bias`` multiplies error rates for
    specific speaker groups.  ``noise_sensitivity`` scales error rates by
    the utterance noise level (lower SNR -> higher rates).
    ``rank_decay`` gives later n-best entries a larger error budget.
    """

    n_utterances: int = 20
    n_best: int = 3
    substitution_rate: float = 0.05
    insertion_rate: float = 0.02
    deletion_rate: float = 0.02
    speaker_groups: tuple[str, ...] = ("A", "B", "C")
    intents: tuple[str, ...] = ("greeting", "qa", "command")
    noise_db_range: tuple[float, float] = (-10.0, 35.0)
    duration_range: tuple[float, float] = (0.5, 12.0)
    seed: int = 0
    confusables: tuple[tuple[str, str], ...] = ()
    group_bias: Mapping[str, float] = field(default_factory=dict)
    noise_sensitivity: float = 0.0
    rank_decay: float = 0.0

    def validate(self) -> None:
        """Raise ``ValueError`` for any inconsistent configuration."""
        if self.n_utterances < 1:
            raise ValueError("n_utterances must be >= 1")
        if self.n_best < 1:
            raise ValueError("n_best must be >= 1")
        if not 0 <= self.substitution_rate <= 1:
            raise ValueError("substitution_rate out of [0,1]")
        if not 0 <= self.insertion_rate <= 1:
            raise ValueError("insertion_rate out of [0,1]")
        if not 0 <= self.deletion_rate <= 1:
            raise ValueError("deletion_rate out of [0,1]")
        total = self.substitution_rate + self.insertion_rate + self.deletion_rate
        if total > 1.0:
            raise ValueError(f"rate sum {total:.4f} exceeds 1.0")
        for key in self.group_bias:
            if key not in self.speaker_groups:
                raise ValueError(f"group_bias key {key!r} not in speaker_groups")
            val = self.group_bias[key]
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"group_bias[{key!r}] must be finite and >= 0")
        if self.rank_decay < 0:
            raise ValueError("rank_decay must be >= 0")
        if not -10.0 <= self.noise_sensitivity <= 10.0:
            raise ValueError("noise_sensitivity out of [-10, 10]")


@dataclass(frozen=True)
class FixtureUtterance:
    """One synthetic utterance with reference, hypotheses and metadata."""

    utterance_id: str
    reference: tuple[str, ...]
    hypotheses: tuple[tuple[str, ...], ...]
    speaker_group: str
    intent_domain: str
    noise_db: float
    duration_s: float
    acoustic_log10s: tuple[float, ...] = ()
    lm_log10s: tuple[float, ...] = ()


_DEFAULT_VOCAB: tuple[str, ...] = (
    "the",
    "cat",
    "sat",
    "on",
    "mat",
    "dog",
    "ran",
    "park",
    "we",
    "saw",
    "hello",
    "world",
    "good",
    "morning",
    "goodbye",
    "evening",
    "yes",
    "no",
    "maybe",
    "please",
    "thanks",
)


def _reference_sentence(
    rng: Any,
    length: int,
    confusables: tuple[tuple[str, str], ...] = (),
) -> list[str]:
    """Generate a random reference sentence of *length* tokens."""
    vocab = [p[0] for p in confusables] if confusables else list(_DEFAULT_VOCAB)
    return [rng.choice(vocab) for _ in range(length)]


def _effective_rates(
    cfg: FixtureConfig,
    group: str | None,
    noise_db: float,
    rank: int,
) -> tuple[float, float, float]:
    """Compute per-token error rates after bias/noise/rank scaling.

    Noise mapping:
        noise_norm = (noise_db - lo) / (hi - lo)   # 0=noisy, 1=clean
        factor = 1 + noise_sensitivity * (1 - noise_norm)
    clamped so factor >= 0.  Rank factor is ``1 + rank_decay * rank``.
    All three rates are clamped to [0, 1] individually.
    """
    sub = cfg.substitution_rate
    ins = cfg.insertion_rate
    dele = cfg.deletion_rate
    if group is not None and group in cfg.group_bias:
        mult = cfg.group_bias[group]
        sub *= mult
        ins *= mult
        dele *= mult
    if cfg.noise_sensitivity != 0.0:
        lo, hi = cfg.noise_db_range
        noise_norm = (noise_db - lo) / (hi - lo) if hi > lo else 0.5
        factor = 1.0 + cfg.noise_sensitivity * (1.0 - noise_norm)
        factor = max(0.0, factor)
        sub *= factor
        ins *= factor
        dele *= factor
    if cfg.rank_decay > 0.0:
        rank_factor = 1.0 + cfg.rank_decay * rank
        sub *= rank_factor
        ins *= rank_factor
        dele *= rank_factor
    return (
        max(0.0, min(sub, 1.0)),
        max(0.0, min(ins, 1.0)),
        max(0.0, min(dele, 1.0)),
    )


def _mutate(
    rng: Any,
    ref: Sequence[str],
    cfg: FixtureConfig,
    group: str | None = None,
    noise_db: float = 0.0,
    rank: int = 0,
) -> tuple[str, ...]:
    """Mutate a reference into a synthetic hypothesis."""
    sub_rate, ins_rate, del_rate = _effective_rates(cfg, group, noise_db, rank)
    out: list[str] = []
    for tok in ref:
        roll = rng.random()
        if roll < sub_rate:
            if cfg.confusables:
                pair = next((p for p in cfg.confusables if p[0] == tok), None)
                if pair is not None:
                    out.append(pair[1])
                else:
                    out.append(rng.choice(cfg.confusables)[1])
            else:
                out.append(rng.choice(["x", "y", "z"]))
        elif roll < sub_rate + ins_rate:
            out.append(tok)
            if cfg.confusables:
                out.append(rng.choice(cfg.confusables)[1])
            else:
                out.append(rng.choice(["x", "y", "z"]))
        elif roll < sub_rate + ins_rate + del_rate:
            continue
        else:
            out.append(tok)
    return tuple(out)


_ACOUSTIC_BASE: float = -10.0
_ACOUSTIC_PENALTY: float = 2.0
_LM_BASE: float = -5.0
_LM_PENALTY: float = 1.5


def _compute_scores(
    ref: tuple[str, ...],
    hyps: tuple[tuple[str, ...], ...],
    noise_db: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Derive acoustic / LM log10 scores from error count.

    Formula (synthetic, not real ASR scoring):
        acoustic = base - penalty * errors - 0.01 * (35 - noise_db)
        lm       = base - penalty * errors
    Fewer errors always yields a higher (less negative) score.
    """
    noise_factor = 0.01 * (35.0 - noise_db)
    acoustic: list[float] = []
    lm: list[float] = []
    for hyp in hyps:
        errors = edit_alignment(ref, hyp).errors
        acoustic.append(_ACOUSTIC_BASE - _ACOUSTIC_PENALTY * errors - noise_factor)
        lm.append(_LM_BASE - _LM_PENALTY * errors)
    return tuple(acoustic), tuple(lm)


def generate_fixture(cfg: FixtureConfig) -> list[FixtureUtterance]:
    """Generate a deterministic list of synthetic fixture utterances."""
    cfg.validate()
    rng = seeded(cfg.seed)
    out: list[FixtureUtterance] = []
    for u in range(cfg.n_utterances):
        length = rng.randint(3, 8)
        ref = tuple(_reference_sentence(rng, length, cfg.confusables))
        group = rng.choice(cfg.speaker_groups)
        intent = rng.choice(cfg.intents)
        noise = rng.uniform(*cfg.noise_db_range)
        duration = rng.uniform(*cfg.duration_range)
        hyps: list[tuple[str, ...]] = []
        for r in range(cfg.n_best):
            hyps.append(_mutate(rng, ref, cfg, group=group, noise_db=noise, rank=r))
        acoustic_log10s, lm_log10s = _compute_scores(ref, tuple(hyps), noise)
        out.append(
            FixtureUtterance(
                utterance_id=f"u{u:04d}",
                reference=ref,
                hypotheses=tuple(hyps),
                speaker_group=group,
                intent_domain=intent,
                noise_db=noise,
                duration_s=duration,
                acoustic_log10s=acoustic_log10s,
                lm_log10s=lm_log10s,
            )
        )
    return out


def as_manifest_dicts(
    fixtures: Iterable[FixtureUtterance],
    config: FixtureConfig | None = None,
) -> list[dict[str, Any]]:
    """Render fixtures as manifest dicts for downstream consumption.

    Always emits one ``hypofuse.system`` row at the end.  When *config*
    is provided its hash is included for reproducibility tracking.
    """
    out: list[dict[str, Any]] = []
    for fx in fixtures:
        out.append(
            {
                "schema": "hypofuse.nbest",
                "utterance_id": fx.utterance_id,
                "system": "synthetic",
                "language": "en",
                "speaker_group": fx.speaker_group,
                "intent_domain": fx.intent_domain,
                "noise_db": fx.noise_db,
                "duration_s": fx.duration_s,
                "hypotheses": [
                    {
                        "rank": i + 1,
                        "text": " ".join(h),
                        "tokens": list(h),
                    }
                    for i, h in enumerate(fx.hypotheses)
                ],
            }
        )
        out.append(
            {
                "schema": "hypofuse.reference",
                "utterance_id": fx.utterance_id,
                "text": " ".join(fx.reference),
                "speaker_group": fx.speaker_group,
                "intent_domain": fx.intent_domain,
                "noise_db": fx.noise_db,
                "duration_s": fx.duration_s,
            }
        )
    config_hash = ""
    if config is not None:
        config_hash = stable_hash(
            {
                "n_utterances": config.n_utterances,
                "n_best": config.n_best,
                "substitution_rate": config.substitution_rate,
                "insertion_rate": config.insertion_rate,
                "deletion_rate": config.deletion_rate,
                "seed": config.seed,
            }
        )
    out.append(
        {
            "schema": "hypofuse.system",
            "system_id": "synthetic",
            "language": "en",
            "config_hash": config_hash,
            "utterance_id": "",
        }
    )
    return out
