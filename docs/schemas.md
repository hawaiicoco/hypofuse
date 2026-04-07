# JSONL Schema Reference

Every manifest record carries a `schema` string and (for alignment and
confusion records) a `schema_version` integer. The package version is
`SCHEMA_VERSION = 1`.

## Schemas

### `hypofuse.nbest`

| Field | Type | Required |
|---|---|---|
| `schema` | string | yes, `"hypofuse.nbest"` |
| `utterance_id` | string (non-empty) | yes |
| `system` | string (non-empty) | yes |
| `language` | string (non-empty) | yes |
| `audio_path` | string | no |
| `hypotheses` | list of objects | no |
| `acoustic_log10` | number | no (validated if present) |
| `lm_log10` | number | no (validated if present) |

Example:
```json
{"schema":"hypofuse.nbest","utterance_id":"u1","system":"a","language":"en","audio_path":"corpus/u1.wav","hypotheses":[{"rank":1,"text":"hi","tokens":["hi"]}]}
```

### `hypofuse.reference`

| Field | Type | Required |
|---|---|---|
| `schema` | string | yes, `"hypofuse.reference"` |
| `utterance_id` | string (non-empty) | yes |
| `text` | string | no |
| `speaker_group` | string | no |
| `duration_s` | number | no |
| `noise_db` | number | no |
| `intent_domain` | string | no |

### `hypofuse.system`

| Field | Type | Required |
|---|---|---|
| `schema` | string | yes, `"hypofuse.system"` |
| `system_id` | string (non-empty) | yes |
| `language` | string (non-empty) | yes |
| `vocabulary_size` | integer | no |
| `config_hash` | string | no |

### `hypofuse.fusion_run`

| Field | Type | Required |
|---|---|---|
| `schema` | string | yes, `"hypofuse.fusion_run"` |
| `utterance_id` | string (non-empty) | yes |
| `policy` | string (non-empty) | yes |
| `systems` | list of strings | no |
| `tokens` | list of strings | no |
| `confidences` | list of numbers | no |
| `config_hash` | string | no |
| `arcs` | list of arc objects | no |

### `hypofuse.report`

| Field | Type | Required |
|---|---|---|
| `schema` | string | yes, `"hypofuse.report"` |
| `kind` | string | no (`"slice"` or `"comparison"`) |
| `key` / `system_a` / `system_b` | string | no |
| `count` / `n_pairs` | integer | no |
| `cer` / `wer` / `delta_cer` / `delta_wer` | number | no |

### `hypofuse.alignment` (used by `alignment_to_jsonl_row`)

| Field | Type | Required |
|---|---|---|
| `schema` | string | yes, `"hypofuse.alignment"` |
| `schema_version` | integer | yes, `1` |
| `utterance_id` | string | yes |
| `system` | string | no |
| `score` | number | yes |
| `ref_length` | integer | yes |
| `hyp_length` | integer | yes |
| `errors` | integer | yes |
| `ops` | list of `{op, ref, hyp}` | yes |

### `hypofuse.confusion` (used by `confusion_to_json`)

| Field | Type | Required |
|---|---|---|
| `schema` | string | yes, `"hypofuse.confusion"` |
| `schema_version` | integer | yes, `1` |
| `slots` | list of slot objects | yes |

Slot: `{"pivot": str, "arcs": [{"token": str, "posterior": number}]}`

## Rejection rules

1. **Unknown schema** -- `validate_record` raises `SchemaError` when
   `schema` is not in `KNOWN_SCHEMAS` (the five core schemas).
2. **Missing required string** -- fields listed for each schema must
   be present, non-empty strings.
3. **Duplicate ids** -- `reject_duplicate_ids` raises `DuplicateIdError`
   when two rows share the same `(schema, utterance_id)` pair. Same id
   across different schemas is allowed.
4. **Audio path escaping** -- `normalize_audio_path` rejects absolute
   paths, NUL bytes, `..` segments, and unsafe characters.
5. **Numeric field types** -- `acoustic_log10` and `lm_log10` must be
   numeric (not boolean) when present.

## Version migration

`SCHEMA_VERSION` is a major version integer. A bump indicates
breaking field changes. Readers should reject records with
`schema_version` values they do not recognize. Currently all schemas
are at version 1.
