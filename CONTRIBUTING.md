# Contributing to hypofuse

Thank you for your interest in contributing to hypofuse.

## Development setup

```bash
# Clone and set up the virtual environment
make install-dev

# Activate the virtual environment
source .venv/bin/activate
```

Requires Python 3.11+ and `uv`.

For neural module development (optional):

```bash
make install-torch
```

## Make targets

| Target | Description |
|---|---|
| `make build` | Build wheel |
| `make test` | Run tests (excludes `slow` and `model` markers) |
| `make test-all` | Run all tests |
| `make format` | Format code with ruff |
| `make format-check` | Check formatting and linting |
| `make lint` | Run ruff linter |
| `make typecheck` | Run mypy type checker |
| `make clean` | Remove build artifacts |
| `make install-dev` | Set up dev environment with uv |
| `make install-torch` | Install torch (CPU) |

## Commit conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.
Scopes: module names (`alignment`, `fusion`, `ngram`, `cli`, etc.).
Subject: imperative mood, lowercase, no period, max 62 characters.

## Test requirements

Every change must pass `make test` and `make format-check`.

### Test style

- One test file per aspect: `tests/test_<module>_<aspect>.py`.
- Docstring on the module, `from __future__ import annotations`.
- Explicit type hints (`-> None`) on test functions.
- Use `pytest.approx` for float comparisons.
- Assert behavior and invariants, not implementation details.
- Golden numbers must be hand-derivable (show arithmetic in comments).
- Synthetic data only, clearly labelled; deterministic seeds.

### Markers

- `@pytest.mark.slow` -- tests taking > ~1 second.
- `@pytest.mark.model` -- tests requiring `torch`.

Default `make test` excludes both markers.

### Golden tests

When adding a test with hard-coded expected values, derive the golden
numbers by hand and show the arithmetic in a comment. This ensures
reviewers can verify the expected output without running the code.

## Adding a new module

1. Add the module to `src/hypofuse/`.
2. Include a module docstring and `from __future__ import annotations`.
3. Use dataclasses with full type hints.
4. Raise typed exceptions from `hypofuse.exceptions`.
5. Add test files to `tests/test_<module>_<aspect>.py`.
6. Update this document and `docs/api.md` if adding public API.

## Adding a new schema

1. Define the dataclass in `src/hypofuse/manifests/`.
2. Add the schema constant to `manifests/__init__.py`.
3. Add required-field validation to `manifests/validate.py`.
4. Add golden tests for serialization roundtrips.
5. Document the schema in `docs/schemas.md`.

## Adding a new CLI subcommand

1. Add an `_add_<name>_args` function and a `_cmd_<name>` handler in `cli.py`.
2. Register in `_build_parser` and `_HANDLERS`.
3. Add the subcommand name to `COMMANDS` and help text to `COMMAND_HELP`.
4. Add CLI tests in `tests/test_cli_<name>.py`.
5. Document in `docs/cli.md`.

## Review checklist

- [ ] All tests pass (`make test`).
- [ ] Format check passes (`make format-check`).
- [ ] New public API is documented in `docs/api.md`.
- [ ] No fabricated numbers or unverified claims.
- [ ] Synthetic data is clearly labelled.
- [ ] Commit message follows conventional commits format.
