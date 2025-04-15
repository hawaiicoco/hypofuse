"""Deterministic synthetic fixture factory.

The factory generates ASR-like data with **known** ground truth so every
metric / fusion / calibration test asserts against a fixed target. All
data is synthetic and labelled as such; nothing here is real ASR output.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from hypofuse.util import seeded


@dataclass(frozen=True)
class FixtureConfig:
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

    def validate(self) -> None:
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


@dataclass(frozen=True)
class FixtureUtterance:
    utterance_id: str
    reference: tuple[str, ...]
    hypotheses: tuple[tuple[str, ...], ...]
    speaker_group: str
    intent_domain: str
    noise_db: float
    duration_s: float
    acoustic_log10s: tuple[float, ...] = ()
    lm_log10s: tuple[float, ...] = ()


def _reference_sentence(rng: Any, length: int) -> list[str]:
    words = [
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
    ]
    return [rng.choice(words) for _ in range(length)]


def _mutate(rng: Any, ref: Sequence[str], cfg: FixtureConfig) -> tuple[str, ...]:
    out: list[str] = []
    for tok in ref:
        roll = rng.random()
        if roll < cfg.substitution_rate:
            out.append(rng.choice(["x", "y", "z"]))
        elif roll < cfg.substitution_rate + cfg.insertion_rate:
            out.append(tok)
            out.append(rng.choice(["x", "y", "z"]))
        elif roll < cfg.substitution_rate + cfg.insertion_rate + cfg.deletion_rate:
            continue
        else:
            out.append(tok)
    return tuple(out)


def generate_fixture(cfg: FixtureConfig) -> list[FixtureUtterance]:
    cfg.validate()
    rng = seeded(cfg.seed)
    out: list[FixtureUtterance] = []
    for u in range(cfg.n_utterances):
        length = rng.randint(3, 8)
        ref = tuple(_reference_sentence(rng, length))
        hyps: list[tuple[str, ...]] = []
        for _ in range(cfg.n_best):
            hyps.append(_mutate(rng, ref, cfg))
        group = rng.choice(cfg.speaker_groups)
        intent = rng.choice(cfg.intents)
        noise = rng.uniform(*cfg.noise_db_range)
        duration = rng.uniform(*cfg.duration_range)
        out.append(
            FixtureUtterance(
                utterance_id=f"u{u:04d}",
                reference=ref,
                hypotheses=tuple(hyps),
                speaker_group=group,
                intent_domain=intent,
                noise_db=noise,
                duration_s=duration,
                acoustic_log10s=tuple(rng.uniform(-50.0, -10.0) for _ in hyps),
                lm_log10s=tuple(rng.uniform(-30.0, -5.0) for _ in hyps),
            )
        )
    return out


def as_manifest_dicts(fixtures: Iterable[FixtureUtterance]) -> list[dict[str, Any]]:
    """Render fixtures as manifest dicts for downstream consumption."""
    out: list[dict[str, Any]] = []
    for fx in fixtures:
        out.append(
            {
                "schema": "hypofuse.nbest",
                "utterance_id": fx.utterance_id,
                "system": "synthetic",
                "language": "en",
                "hypotheses": [
                    {"rank": i + 1, "text": " ".join(h), "tokens": list(h)}
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
    return out
