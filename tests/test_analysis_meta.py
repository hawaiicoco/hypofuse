"""Report metadata block in markdown output."""

from __future__ import annotations

from hypofuse.analysis import SliceMetric, report_to_markdown


def test_meta_renders_sorted_keys() -> None:
    md = report_to_markdown(
        [SliceMetric("all", 5, 0.1, 0.2)],
        [],
        meta={"corpus": "test", "seed": 42},
    )
    assert "## Run" in md
    lines = md.split("\n")
    run_idx = next(i for i, ln in enumerate(lines) if ln == "## Run")
    key_lines = [ln for ln in lines[run_idx + 1 :] if ": " in ln and not ln.startswith("|")]
    assert key_lines[0].startswith("corpus:")
    assert key_lines[1].startswith("seed:")


def test_omitting_meta_preserves_output() -> None:
    md = report_to_markdown([SliceMetric("all", 5, 0.1, 0.2)], [])
    assert "## Run" not in md
    expected_lines = [
        "# Error Analysis Report",
        "",
        "## Slices",
        "| Slice | Count | CER | WER |",
        "|---|---|---|---|",
        "| all | 5 | 0.1000 | 0.2000 |",
        "",
        "## System comparisons (paired bootstrap, 95% CI)",
        "| Pair | n | \u0394 CER | CER CI | \u0394 WER | WER CI |",
        "|---|---|---|---|---|---|",
    ]
    assert md == "\n".join(expected_lines)


def test_meta_none_default() -> None:
    md1 = report_to_markdown([SliceMetric("x", 1, 0.0, 0.0)], [])
    md2 = report_to_markdown([SliceMetric("x", 1, 0.0, 0.0)], [], meta=None)
    assert md1 == md2
