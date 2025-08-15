"""Golden test for RunMetadata JSON serialization.

Replaces machine-specific fields with placeholders before comparison so the
golden does not drift between machines.
"""

from __future__ import annotations

import json

from hypofuse.runmeta import RunMetadata, to_json


def test_golden_metadata_json() -> None:
    """Verify the JSON structure matches a golden template.

    Machine-specific fields (python_version, numpy_version, platform,
    torch_version) are replaced with placeholders before comparison.
    """
    meta = RunMetadata(
        hypofuse_version="0.1.0",
        python_version="3.11.5",
        platform="linux",
        numpy_version="1.26.0",
        seed=42,
        config_hash="abc123",
        created_from="test:golden",
        torch_version=None,
        stamped_at=None,
    )
    text = to_json(meta)
    data = json.loads(text)
    data["python_version"] = "<PYTHON_VERSION>"
    data["numpy_version"] = "<NUMPY_VERSION>"
    data["platform"] = "<PLATFORM>"
    data["torch_version"] = "<TORCH_VERSION>"
    normalized = json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)
    expected = {
        "config_hash": "abc123",
        "created_from": "test:golden",
        "hypofuse_version": "0.1.0",
        "numpy_version": "<NUMPY_VERSION>",
        "platform": "<PLATFORM>",
        "python_version": "<PYTHON_VERSION>",
        "seed": 42,
        "stamped_at": None,
        "torch_version": "<TORCH_VERSION>",
    }
    expected_text = json.dumps(expected, sort_keys=True, indent=2, ensure_ascii=False)
    assert normalized == expected_text
