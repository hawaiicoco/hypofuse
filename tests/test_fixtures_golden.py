"""Golden manifest text to catch schema drift.

The golden string was generated from
``FixtureConfig(n_utterances=2, n_best=2, seed=7)``
using ``json.dumps(rows, indent=2, sort_keys=True)``.
Any deliberate schema change requires updating this golden.
"""

from __future__ import annotations

import json

from hypofuse.fixtures import (
    FixtureConfig,
    as_manifest_dicts,
    generate_fixture,
)

# fmt: off
_GOLDEN = """[
  {
    "duration_s": 1.1669876349091282,
    "hypotheses": [
      {
        "rank": 1,
        "text": "mat y thanks cat sat",
        "tokens": [
          "mat",
          "y",
          "thanks",
          "cat",
          "sat"
        ]
      },
      {
        "rank": 2,
        "text": "mat z good thanks cat sat",
        "tokens": [
          "mat",
          "z",
          "good",
          "thanks",
          "cat",
          "sat"
        ]
      }
    ],
    "intent_domain": "greeting",
    "language": "en",
    "noise_db": 6.456001261066348,
    "schema": "hypofuse.nbest",
    "speaker_group": "C",
    "system": "synthetic",
    "utterance_id": "u0000"
  },
  {
    "duration_s": 1.1669876349091282,
    "intent_domain": "greeting",
    "noise_db": 6.456001261066348,
    "schema": "hypofuse.reference",
    "speaker_group": "C",
    "text": "mat good thanks cat sat",
    "utterance_id": "u0000"
  },
  {
    "duration_s": 1.8546107379012362,
    "hypotheses": [
      {
        "rank": 1,
        "text": "maybe good cat park cat no mat",
        "tokens": [
          "maybe",
          "good",
          "cat",
          "park",
          "cat",
          "no",
          "mat"
        ]
      },
      {
        "rank": 2,
        "text": "maybe x good cat park cat no mat",
        "tokens": [
          "maybe",
          "x",
          "good",
          "cat",
          "park",
          "cat",
          "no",
          "mat"
        ]
      }
    ],
    "intent_domain": "qa",
    "language": "en",
    "noise_db": -3.5085212489153115,
    "schema": "hypofuse.nbest",
    "speaker_group": "B",
    "system": "synthetic",
    "utterance_id": "u0001"
  },
  {
    "duration_s": 1.8546107379012362,
    "intent_domain": "qa",
    "noise_db": -3.5085212489153115,
    "schema": "hypofuse.reference",
    "speaker_group": "B",
    "text": "maybe good cat park cat no mat",
    "utterance_id": "u0001"
  },
  {
    "config_hash": "2b4890d31cfcb1991da9fa3786328ce5a9271e288613e2ad3f8fc80ce67e43cb",
    "language": "en",
    "schema": "hypofuse.system",
    "system_id": "synthetic",
    "utterance_id": ""
  }
]"""
# fmt: on


def test_golden_manifest_text() -> None:
    cfg = FixtureConfig(n_utterances=2, n_best=2, seed=7)
    fx = generate_fixture(cfg)
    rows = as_manifest_dicts(fx, config=cfg)
    actual = json.dumps(rows, indent=2, sort_keys=True)
    assert actual == _GOLDEN
