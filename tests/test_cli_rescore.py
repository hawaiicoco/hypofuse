"""CLI tests for the rescore subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypofuse.cli import main
from hypofuse.fixtures import FixtureConfig, as_manifest_dicts, generate_fixture
from hypofuse.manifests.jsonl import write_manifest
from hypofuse.ngram import NgramLM


def _write_nbest(tmp_path: Path) -> Path:
    fx = generate_fixture(FixtureConfig(n_utterances=2, n_best=3, seed=3))
    rows = [r for r in as_manifest_dicts(fx) if r["schema"] == "hypofuse.nbest"]
    path = tmp_path / "nbest.jsonl"
    write_manifest(path, rows)
    return path


def _write_arpa(tmp_path: Path) -> Path:
    lm = NgramLM.train([["hello", "world"], ["good", "morning"]], order=2)
    path = tmp_path / "lm.arpa"
    path.write_text(lm.to_arpa(), encoding="utf-8")
    return path


def _write_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text("hello world\ngood morning\n", encoding="utf-8")
    return path


def test_rescore_arpa(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nb = _write_nbest(tmp_path)
    arpa = _write_arpa(tmp_path)
    code = main(["rescore", "--nbest", str(nb), "--arpa", str(arpa), "--top", "2"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list)
    assert len(data) >= 1
    assert len(data[0]["hypotheses"]) <= 2


def test_rescore_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nb = _write_nbest(tmp_path)
    corpus = _write_corpus(tmp_path)
    code = main(
        [
            "rescore",
            "--nbest",
            str(nb),
            "--corpus",
            str(corpus),
            "--order",
            "2",
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list)


def test_rescore_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["rescore", "--nbest", "/no/such.jsonl", "--arpa", "/x.arpa"])
    assert code == 2
    assert "hypofuse rescore" in capsys.readouterr().err
