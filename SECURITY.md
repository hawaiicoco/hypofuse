# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |

Only the latest release in each minor series receives security updates.

## Reporting a vulnerability

Please report security issues privately by emailing the maintainer
at the address listed in `pyproject.toml` (Author field). Do not
open a public GitHub issue for security vulnerabilities.

Include:
- A description of the vulnerability.
- Steps to reproduce.
- The affected version(s).
- Any suggested fix or mitigation.

We aim to acknowledge receipt within 5 business days and provide
an initial assessment within 15 business days.

## Scope

hypofuse is an **offline research toolkit** with no network surface:

- The CLI operates on local files only.
- There is no web server, API endpoint, or network communication.
- The `demo` subcommand generates synthetic data locally.
- No data is transmitted to external services.

## Path-traversal safety

Manifest audio paths are validated by `normalize_audio_path` and
`safe_audio_join` in `hypofuse.manifests.paths`:

- Absolute paths are rejected.
- Parent-directory escapes (`..`) are rejected.
- NUL bytes are rejected.
- Unsafe characters in path segments are rejected.
- `safe_audio_join` verifies the resolved path stays within the base
  directory.

These checks prevent path-traversal attacks when processing manifests
from untrusted sources.

## Untrusted JSONL input

The JSONL reader (`read_jsonl` / `read_manifest`) uses Python's
standard `json.loads` which does not execute arbitrary code. Schema
validation (`validate_record`) checks field types and required fields
before any downstream processing.

**Recommendation:** When processing manifests from untrusted sources,
validate them with `hypofuse validate` before running analysis or
fusion commands.

## Dependencies

hypofuse depends on:
- **numpy** (required): numerical computation.
- **torch** (optional): only when the neural module is used.

Both are well-maintained open-source projects. Pin your dependency
versions and monitor upstream security advisories.
