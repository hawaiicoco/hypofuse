"""Strict manifest row <-> dataclass conversion.

Every ``*_from_row`` function accepts a plain dict (as read from JSONL)
and returns the corresponding frozen dataclass with strict type checks.
Unknown keys are rejected by default (``allow_extra=False``).  Missing
optional keys are filled from dataclass defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import hypofuse.manifests as _m
from hypofuse.exceptions import SchemaError
from hypofuse.manifests import (
    SCHEMA_FUSION_RUN,
    SCHEMA_NBEST,
    SCHEMA_REFERENCE,
    SCHEMA_REPORT,
    SCHEMA_SYSTEM,
)
from hypofuse.manifests.fusion_run import FusionArc, FusionRun
from hypofuse.manifests.nbest import NBestHypothesis, NBestList
from hypofuse.manifests.reference import ReferenceTranscript
from hypofuse.manifests.reporter import ReportRecord
from hypofuse.manifests.system import SystemMetadata

_META_KEYS = frozenset({"schema", "schema_version"})


def _require(row: Mapping[str, Any], key: str, expected: type) -> Any:
    """Return *row[key]* after asserting presence and exact type."""
    if key not in row:
        raise SchemaError(f"missing required field: {key}")
    val = row[key]
    if expected is int:
        if not isinstance(val, int) or isinstance(val, bool):
            raise SchemaError(f"field {key} expected int, got {type(val).__name__}")
    elif expected is float:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise SchemaError(f"field {key} expected float, got {type(val).__name__}")
    elif not isinstance(val, expected):
        raise SchemaError(f"field {key} expected {expected.__name__}, got {type(val).__name__}")
    return val


def _optional(row: Mapping[str, Any], key: str, expected: type, default: Any) -> Any:
    """Return *row[key]* with type check, or *default* when absent."""
    if key not in row:
        return default
    val = row[key]
    if val is None and default is None:
        return None
    if expected is int:
        if not isinstance(val, int) or isinstance(val, bool):
            raise SchemaError(f"field {key} expected int, got {type(val).__name__}")
    elif expected is float:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise SchemaError(f"field {key} expected float, got {type(val).__name__}")
        return float(val)
    elif not isinstance(val, expected):
        raise SchemaError(f"field {key} expected {expected.__name__}, got {type(val).__name__}")
    return val


def _reject_extra(row: Mapping[str, Any], known: frozenset[str]) -> None:
    extra = set(row.keys()) - known - _META_KEYS
    if extra:
        raise SchemaError(f"unknown keys: {sorted(extra)}")


_NBEST_KEYS = frozenset({"utterance_id", "system", "language", "hypotheses", "audio_path"})

_HYP_KEYS = frozenset(
    {
        "rank",
        "text",
        "tokens",
        "posteriors",
        "acoustic_log10",
        "lm_log10",
        "start_time",
        "end_time",
    }
)


def _hyp_from_dict(d: Any, *, idx: int) -> NBestHypothesis:
    if not isinstance(d, dict):
        raise SchemaError(f"hypotheses[{idx}] expected dict, got {type(d).__name__}")
    extra = set(d.keys()) - _HYP_KEYS
    if extra:
        raise SchemaError(f"hypotheses[{idx}]: unknown keys: {sorted(extra)}")
    rank = _require(d, "rank", int)
    text = _require(d, "text", str)
    tokens_raw = _optional(d, "tokens", list, [])
    for i, t in enumerate(tokens_raw):
        if not isinstance(t, str):
            raise SchemaError(f"hypotheses[{idx}].tokens[{i}] expected str")
    post_raw = _optional(d, "posteriors", list, [])
    for i, p in enumerate(post_raw):
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise SchemaError(f"hypotheses[{idx}].posteriors[{i}] expected float")
    return NBestHypothesis(
        rank=rank,
        text=text,
        tokens=tuple(tokens_raw),
        posteriors=tuple(float(p) for p in post_raw),
        acoustic_log10=_optional(d, "acoustic_log10", float, 0.0),
        lm_log10=_optional(d, "lm_log10", float, 0.0),
        start_time=_optional(d, "start_time", float, None),
        end_time=_optional(d, "end_time", float, None),
    )


def nbest_from_row(row: Mapping[str, Any], *, allow_extra: bool = False) -> NBestList:
    """Convert a ``hypofuse.nbest`` dict row to an NBestList."""
    if not isinstance(row, Mapping):
        raise SchemaError(f"expected dict, got {type(row).__name__}")
    schema = row.get("schema")
    if schema is not None and schema != SCHEMA_NBEST:
        raise SchemaError(f"expected schema {SCHEMA_NBEST!r}, got {schema!r}")
    if not allow_extra:
        _reject_extra(row, _NBEST_KEYS)
    uid = _require(row, "utterance_id", str)
    system = _require(row, "system", str)
    language = _require(row, "language", str)
    hyps_raw = _optional(row, "hypotheses", list, [])
    hypotheses = tuple(_hyp_from_dict(h, idx=i) for i, h in enumerate(hyps_raw))
    audio_path = _optional(row, "audio_path", str, "")
    return NBestList(
        utterance_id=uid,
        system=system,
        language=language,
        hypotheses=hypotheses,
        audio_path=audio_path,
    )


def nbest_to_row(obj: NBestList) -> dict[str, Any]:
    """Convert an NBestList to a JSON-serializable dict."""
    return {
        "schema": SCHEMA_NBEST,
        "schema_version": _m.SCHEMA_VERSIONS[SCHEMA_NBEST],
        "utterance_id": obj.utterance_id,
        "system": obj.system,
        "language": obj.language,
        "hypotheses": [
            {
                "rank": h.rank,
                "text": h.text,
                "tokens": list(h.tokens),
                "posteriors": list(h.posteriors),
                "acoustic_log10": h.acoustic_log10,
                "lm_log10": h.lm_log10,
                "start_time": h.start_time,
                "end_time": h.end_time,
            }
            for h in obj.hypotheses
        ],
        "audio_path": obj.audio_path,
    }


_REF_KEYS = frozenset(
    {
        "utterance_id",
        "text",
        "speaker_id",
        "speaker_group",
        "duration_s",
        "noise_db",
        "intent_domain",
        "tokens",
    }
)


def reference_from_row(row: Mapping[str, Any], *, allow_extra: bool = False) -> ReferenceTranscript:
    """Convert a ``hypofuse.reference`` dict row to a ReferenceTranscript."""
    if not isinstance(row, Mapping):
        raise SchemaError(f"expected dict, got {type(row).__name__}")
    if not allow_extra:
        _reject_extra(row, _REF_KEYS)
    uid = _require(row, "utterance_id", str)
    text = _require(row, "text", str)
    tokens_raw = _optional(row, "tokens", list, [])
    for i, t in enumerate(tokens_raw):
        if not isinstance(t, str):
            raise SchemaError(f"tokens[{i}] expected str")
    return ReferenceTranscript(
        utterance_id=uid,
        text=text,
        speaker_id=_optional(row, "speaker_id", str, ""),
        speaker_group=_optional(row, "speaker_group", str, ""),
        duration_s=_optional(row, "duration_s", float, 0.0),
        noise_db=_optional(row, "noise_db", float, 0.0),
        intent_domain=_optional(row, "intent_domain", str, ""),
        tokens=tuple(tokens_raw),
    )


def reference_to_row(obj: ReferenceTranscript) -> dict[str, Any]:
    """Convert a ReferenceTranscript to a JSON-serializable dict."""
    return {
        "schema": SCHEMA_REFERENCE,
        "schema_version": _m.SCHEMA_VERSIONS[SCHEMA_REFERENCE],
        "utterance_id": obj.utterance_id,
        "text": obj.text,
        "speaker_id": obj.speaker_id,
        "speaker_group": obj.speaker_group,
        "duration_s": obj.duration_s,
        "noise_db": obj.noise_db,
        "intent_domain": obj.intent_domain,
        "tokens": list(obj.tokens),
    }


_SYS_KEYS = frozenset(
    {
        "system_id",
        "language",
        "vocabulary_size",
        "description",
        "acoustic_model",
        "language_model",
        "decoder",
        "version",
    }
)


def system_from_row(row: Mapping[str, Any], *, allow_extra: bool = False) -> SystemMetadata:
    """Convert a ``hypofuse.system`` dict row to SystemMetadata."""
    if not isinstance(row, Mapping):
        raise SchemaError(f"expected dict, got {type(row).__name__}")
    if not allow_extra:
        _reject_extra(row, _SYS_KEYS)
    sid = _require(row, "system_id", str)
    lang = _require(row, "language", str)
    return SystemMetadata(
        system_id=sid,
        language=lang,
        vocabulary_size=_optional(row, "vocabulary_size", int, 0),
        description=_optional(row, "description", str, ""),
        acoustic_model=_optional(row, "acoustic_model", str, ""),
        language_model=_optional(row, "language_model", str, ""),
        decoder=_optional(row, "decoder", str, ""),
        version=_optional(row, "version", str, ""),
    )


def system_to_row(obj: SystemMetadata) -> dict[str, Any]:
    """Convert SystemMetadata to a JSON-serializable dict."""
    return {
        "schema": SCHEMA_SYSTEM,
        "schema_version": _m.SCHEMA_VERSIONS[SCHEMA_SYSTEM],
        "system_id": obj.system_id,
        "language": obj.language,
        "vocabulary_size": obj.vocabulary_size,
        "description": obj.description,
        "acoustic_model": obj.acoustic_model,
        "language_model": obj.language_model,
        "decoder": obj.decoder,
        "version": obj.version,
    }


_FUSION_KEYS = frozenset(
    {
        "utterance_id",
        "systems",
        "tokens",
        "confidences",
        "policy",
        "config_hash",
        "arcs",
    }
)

_ARC_KEYS = frozenset({"pivot", "candidates"})


def _arc_from_dict(d: Any, *, idx: int) -> FusionArc:
    if not isinstance(d, dict):
        raise SchemaError(f"arcs[{idx}] expected dict, got {type(d).__name__}")
    extra = set(d.keys()) - _ARC_KEYS
    if extra:
        raise SchemaError(f"arcs[{idx}]: unknown keys: {sorted(extra)}")
    pivot = _require(d, "pivot", str)
    cands_raw = _optional(d, "candidates", list, [])
    candidates: list[tuple[str, float]] = []
    for j, c in enumerate(cands_raw):
        if not isinstance(c, (list, tuple)) or len(c) != 2:
            raise SchemaError(f"arcs[{idx}].candidates[{j}] expected [str, float] pair")
        tok, prob = c
        if not isinstance(tok, str):
            raise SchemaError(f"arcs[{idx}].candidates[{j}][0] expected str")
        if isinstance(prob, bool) or not isinstance(prob, (int, float)):
            raise SchemaError(f"arcs[{idx}].candidates[{j}][1] expected float")
        candidates.append((tok, float(prob)))
    return FusionArc(pivot=pivot, candidates=tuple(candidates))


def fusion_run_from_row(row: Mapping[str, Any], *, allow_extra: bool = False) -> FusionRun:
    """Convert a ``hypofuse.fusion_run`` dict row to a FusionRun."""
    if not isinstance(row, Mapping):
        raise SchemaError(f"expected dict, got {type(row).__name__}")
    if not allow_extra:
        _reject_extra(row, _FUSION_KEYS)
    uid = _require(row, "utterance_id", str)
    systems_raw = _require(row, "systems", list)
    for i, s in enumerate(systems_raw):
        if not isinstance(s, str):
            raise SchemaError(f"systems[{i}] expected str")
    tokens_raw = _require(row, "tokens", list)
    for i, t in enumerate(tokens_raw):
        if not isinstance(t, str):
            raise SchemaError(f"tokens[{i}] expected str")
    conf_raw = _require(row, "confidences", list)
    for i, c in enumerate(conf_raw):
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            raise SchemaError(f"confidences[{i}] expected float")
    policy = _require(row, "policy", str)
    arcs_raw = _optional(row, "arcs", list, [])
    arcs = tuple(_arc_from_dict(a, idx=i) for i, a in enumerate(arcs_raw))
    return FusionRun(
        utterance_id=uid,
        systems=tuple(systems_raw),
        tokens=tuple(tokens_raw),
        confidences=tuple(float(c) for c in conf_raw),
        policy=policy,
        config_hash=_optional(row, "config_hash", str, ""),
        arcs=arcs,
    )


def fusion_run_to_row(obj: FusionRun) -> dict[str, Any]:
    """Convert a FusionRun to a JSON-serializable dict."""
    return {
        "schema": SCHEMA_FUSION_RUN,
        "schema_version": _m.SCHEMA_VERSIONS[SCHEMA_FUSION_RUN],
        "utterance_id": obj.utterance_id,
        "systems": list(obj.systems),
        "tokens": list(obj.tokens),
        "confidences": list(obj.confidences),
        "policy": obj.policy,
        "config_hash": obj.config_hash,
        "arcs": [
            {
                "pivot": a.pivot,
                "candidates": [[tok, prob] for tok, prob in a.candidates],
            }
            for a in obj.arcs
        ],
    }


_REPORT_KEYS = frozenset(
    {
        "report_id",
        "schema",
        "generated_at",
        "hypofuse_version",
        "config_hash",
        "seed",
        "artifact_paths",
        "metrics",
    }
)


def report_from_row(row: Mapping[str, Any], *, allow_extra: bool = False) -> ReportRecord:
    """Convert a ``hypofuse.report`` dict row to a ReportRecord."""
    if not isinstance(row, Mapping):
        raise SchemaError(f"expected dict, got {type(row).__name__}")
    if not allow_extra:
        _reject_extra(row, _REPORT_KEYS)
    rid = _require(row, "report_id", str)
    schema_val = _require(row, "schema", str)
    generated_at = _require(row, "generated_at", str)
    version = _require(row, "hypofuse_version", str)
    config_hash = _require(row, "config_hash", str)
    seed = _require(row, "seed", int)
    paths_raw = _optional(row, "artifact_paths", list, [])
    for i, p in enumerate(paths_raw):
        if not isinstance(p, str):
            raise SchemaError(f"artifact_paths[{i}] expected str")
    metrics_raw = _optional(row, "metrics", list, [])
    metrics: list[tuple[str, float]] = []
    for i, m in enumerate(metrics_raw):
        if not isinstance(m, (list, tuple)) or len(m) != 2:
            raise SchemaError(f"metrics[{i}] expected [str, float] pair")
        name, val = m
        if not isinstance(name, str):
            raise SchemaError(f"metrics[{i}][0] expected str")
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise SchemaError(f"metrics[{i}][1] expected float")
        metrics.append((name, float(val)))
    return ReportRecord(
        report_id=rid,
        schema=schema_val,
        generated_at=generated_at,
        hypofuse_version=version,
        config_hash=config_hash,
        seed=seed,
        artifact_paths=tuple(paths_raw),
        metrics=tuple(metrics),
    )


def report_to_row(obj: ReportRecord) -> dict[str, Any]:
    """Convert a ReportRecord to a JSON-serializable dict."""
    return {
        "schema": obj.schema,
        "schema_version": _m.SCHEMA_VERSIONS[SCHEMA_REPORT],
        "report_id": obj.report_id,
        "generated_at": obj.generated_at,
        "hypofuse_version": obj.hypofuse_version,
        "config_hash": obj.config_hash,
        "seed": obj.seed,
        "artifact_paths": list(obj.artifact_paths),
        "metrics": [[name, val] for name, val in obj.metrics],
    }


_SCHEMA_DISPATCH: dict[str, Any] = {
    SCHEMA_NBEST: nbest_from_row,
    SCHEMA_REFERENCE: reference_from_row,
    SCHEMA_SYSTEM: system_from_row,
    SCHEMA_FUSION_RUN: fusion_run_from_row,
    SCHEMA_REPORT: report_from_row,
}


def convert_row(row: Mapping[str, Any], *, allow_extra: bool = False) -> Any:
    """Dispatch to the correct ``*_from_row`` based on ``row["schema"]``.

    Raises SchemaError listing supported schemas for an unknown one.
    """
    if not isinstance(row, Mapping):
        raise SchemaError(f"expected dict, got {type(row).__name__}")
    schema = row.get("schema")
    if schema is None:
        raise SchemaError("missing schema field")
    handler = _SCHEMA_DISPATCH.get(schema)
    if handler is None:
        supported = sorted(_SCHEMA_DISPATCH.keys())
        raise SchemaError(f"unknown schema {schema!r}; supported: {supported}")
    return handler(row, allow_extra=allow_extra)
