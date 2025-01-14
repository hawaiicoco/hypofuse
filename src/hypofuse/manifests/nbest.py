"""N-best hypothesis lists."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NBestHypothesis:
    """A single ranked hypothesis for one utterance.

    `tokens` are positional strings (word for English, char for Chinese
    scoring). `posteriors` are optional per-token probabilities.
    Acoustic and LM log-scores are stored as log10 to keep the values
    human-readable and comparable with open-source toolkits.
    """

    rank: int
    text: str
    tokens: tuple[str, ...] = ()
    posteriors: tuple[float, ...] = ()
    acoustic_log10: float = 0.0
    lm_log10: float = 0.0
    start_time: float | None = None
    end_time: float | None = None


@dataclass(frozen=True)
class NBestList:
    """A ranked list of hypotheses for one utterance."""

    utterance_id: str
    system: str
    language: str
    hypotheses: tuple[NBestHypothesis, ...] = ()
    audio_path: str = ""

    def ranks(self) -> tuple[int, ...]:
        return tuple(h.rank for h in self.hypotheses)
