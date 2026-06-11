# Reproducibility: Seeds, Locks, and Artifacts

## Seeded generators

All randomness in hypofuse flows through `hypofuse.util.seeded(seed)`,
which returns a `random.Random` instance seeded with `version=2`
(Mersenne Twister). No module touches `random.random()` or
`random.seed()` globally.

The fixture generator (`generate_fixture`), bootstrap CI
(`paired_bootstrap_ci`), and p-value estimator
(`paired_bootstrap_p_value`) all accept a `seed` parameter.
Same seed + same input = identical output, byte-for-byte.

## Pinned lockfile

The project uses `uv` for dependency management. The development
environment is set up with:

```
make install-dev
```

This runs `uv venv .venv --python python3.11` and
`uv pip install -e ".[dev]"`. Dependencies are pinned by the
`pyproject.toml` requirements:

| Dependency | Version |
|---|---|
| numpy | >= 1.24 |
| pytest | >= 8.0 |
| pytest-cov | >= 5.0 |
| ruff | >= 0.6 |
| mypy | >= 1.10 |
| torch (optional) | >= 2.2 |

## Versioned artifacts

JSONL manifests carry `schema` and `schema_version` fields.
The schema version is `SCHEMA_VERSION = 1`. Readers should reject
records with unrecognized schema versions.

`RunMetadata` captures: `hypofuse_version`, `python_version`,
`platform`, `numpy_version`, `seed`, `config_hash`, and optionally
`torch_version` and `stamped_at`.

## Run metadata contents

```python
from hypofuse.runmeta import capture, to_json
meta = capture(seed=42, config=my_config, label="experiment-1")
print(to_json(meta))
```

Output includes:
```json
{
  "config_hash": "abc123...",
  "created_from": "experiment-1",
  "hypofuse_version": "0.1.0",
  "numpy_version": "1.26.0",
  "platform": "linux",
  "python_version": "3.11.9",
  "seed": 42,
  "stamped_at": null,
  "torch_version": null
}
```

`stamp(meta, when)` adds a caller-supplied ISO-8601 timestamp
without reading wall-clock time, preserving reproducibility.

## Reproducing a report byte-for-byte

1. Use the same `seed` for fixture generation and bootstrap.
2. Use the same `FixtureConfig` (its hash is embedded in the system row).
3. Use the same Python version and hypofuse version.
4. Write output via `write_jsonl` (deterministic: sorted keys,
   UTF-8, LF line endings).

```python
from hypofuse.fixtures import FixtureConfig, generate_fixture, as_manifest_dicts
from hypofuse.util import write_jsonl

cfg = FixtureConfig(n_utterances=10, seed=42)
fixtures = generate_fixture(cfg)
rows = as_manifest_dicts(fixtures, config=cfg)
write_jsonl("output.jsonl", rows)
```

Running this script twice with the same hypofuse version produces
byte-identical output.

## Test markers

| Marker | Purpose | Typical runtime |
|---|---|---|
| (none) | Default test suite | < 1 s per test |
| `@pytest.mark.slow` | Tests with > ~1 s runtime | Up to 60 s |
| `@pytest.mark.model` | Tests requiring `torch` | Variable |

Default `make test` runs `pytest -q -m "not slow and not model"`,
completing in approximately 15 seconds. `make test-all` runs
everything including slow and model tests.

Slow tests include:
- Large banded alignment (2000 tokens, band=200).
- Large bootstrap (500 utterances, 5000 iterations).
- Confusion stress (500 grids, 8 hypotheses each).
- Fixture scale (2000 utterances).

Model tests require `pip install hypofuse[torch]` and are skipped
when torch is not available.

## Release process

hypofuse uses patch-only versioning (`0.1.N`).

1. Bump version in `pyproject.toml` and `src/hypofuse/_version.py`.
2. Add `## [0.1.N] - YYYY-MM` to `CHANGELOG.md`.
3. Run `make release`.
4. Tag `v0.1.N` and push.
5. GitHub Release attaches wheel/sdist. No PyPI publishing.

`tests/test_release.py` enforces version consistency.
