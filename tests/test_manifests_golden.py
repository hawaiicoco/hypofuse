"""Golden JSON shape for every manifest schema.

Each schema ships a tiny golden record. Tests serialize the dataclass via
``to_dict`` helpers and assert byte-stable JSON output. This makes schema
drift loud: any change to field names, ordering or required fields fails.
"""

from __future__ import annotations

import json

from hypofuse.manifests.fusion_run import FusionArc, FusionRun
from hypofuse.manifests.nbest import NBestHypothesis, NBestList
from hypofuse.manifests.reference import ReferenceTranscript
from hypofuse.manifests.reporter import ReportRecord
from hypofuse.manifests.system import SystemMetadata
from hypofuse.util import json_default


def _dump(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=json_default)


def test_golden_nbest() -> None:
    nb = NBestList(
        utterance_id="u1",
        system="a",
        language="en",
        hypotheses=(NBestHypothesis(rank=1, text="hi", tokens=("hi",)),),
        audio_path="corpus/u1.wav",
    )
    assert _dump(nb) == json.dumps(
        {
            "utterance_id": "u1",
            "system": "a",
            "language": "en",
            "hypotheses": [
                {
                    "rank": 1,
                    "text": "hi",
                    "tokens": ["hi"],
                    "posteriors": [],
                    "acoustic_log10": 0.0,
                    "lm_log10": 0.0,
                    "start_time": None,
                    "end_time": None,
                }
            ],
            "audio_path": "corpus/u1.wav",
        },
        sort_keys=True,
    )


def test_golden_reference() -> None:
    ref = ReferenceTranscript(
        utterance_id="u1",
        text="hi",
        speaker_id="spk1",
        speaker_group="A",
        duration_s=1.2,
        noise_db=10.0,
        intent_domain="greeting",
    )
    payload = json.loads(_dump(ref))
    assert payload == {
        "utterance_id": "u1",
        "text": "hi",
        "speaker_id": "spk1",
        "speaker_group": "A",
        "duration_s": 1.2,
        "noise_db": 10.0,
        "intent_domain": "greeting",
        "tokens": [],
    }


def test_golden_system() -> None:
    sys_md = SystemMetadata(system_id="a", language="en", vocabulary_size=10)
    payload = json.loads(_dump(sys_md))
    assert payload["system_id"] == "a"
    assert payload["language"] == "en"
    assert payload["vocabulary_size"] == 10


def test_golden_fusion_run() -> None:
    fr = FusionRun(
        utterance_id="u1",
        systems=("a", "b"),
        tokens=("hi",),
        confidences=(0.7,),
        policy="majority",
        arcs=(FusionArc(pivot="*", candidates=(("hi", 1.0),)),),
    )
    payload = json.loads(_dump(fr))
    assert payload["policy"] == "majority"
    assert payload["tokens"] == ["hi"]
    assert payload["arcs"][0]["pivot"] == "*"


def test_golden_report() -> None:
    r = ReportRecord(
        report_id="rep_1",
        schema="hypofuse.report",
        generated_at="2026-09-26T00:00:00Z",
        hypofuse_version="0.1.0",
        config_hash="abc",
        seed=42,
        metrics=(("wer", 0.1),),
    )
    payload = json.loads(_dump(r))
    assert payload["seed"] == 42
    assert payload["metrics"] == [["wer", 0.1]]
