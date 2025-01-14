"""Reference transcripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceTranscript:
    """A single reference transcript with optional segmentation metadata.

    `tokens` defaults to an empty tuple; populating it lets callers run
    alignment without re-segmenting the text.
    """

    utterance_id: str
    text: str
    speaker_id: str = ""
    speaker_group: str = ""
    duration_s: float = 0.0
    noise_db: float = 0.0
    intent_domain: str = ""
    tokens: tuple[str, ...] = ()
