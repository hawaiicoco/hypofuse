"""Example ex01: validate manifests and score synthetic fixtures.

Fully offline, deterministic and synthetic.  No network access, no real
ASR output.  All data is generated in-process with seeded generators.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from hypofuse.alignment import character_error_rate, word_error_rate
from hypofuse.exceptions import SchemaError
from hypofuse.fixtures import FixtureConfig, as_manifest_dicts, generate_fixture
from hypofuse.manifests.jsonl import read_manifest, write_manifest
from hypofuse.normalize import NormalizationConfig, normalize_pair


def main(outdir: str) -> int:
    """Run the ex01 example, writing artifacts into *outdir*."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)

    # 1. Generate a small synthetic fixture.
    cfg = FixtureConfig(n_utterances=3, n_best=2, seed=42)
    fixtures = generate_fixture(cfg)
    rows = as_manifest_dicts(fixtures, config=cfg)

    nbest_rows = [r for r in rows if r["schema"] == "hypofuse.nbest"]
    ref_rows = [r for r in rows if r["schema"] == "hypofuse.reference"]

    # 2. Write JSONL manifests.
    write_manifest(target / "nbest.jsonl", nbest_rows)
    write_manifest(target / "reference.jsonl", ref_rows)

    # 3. Validate by reading them back.
    loaded_nbest = read_manifest(target / "nbest.jsonl")
    loaded_ref = read_manifest(target / "reference.jsonl")
    print(f"validated {len(loaded_nbest)} n-best rows and {len(loaded_ref)} reference rows")

    # 4. Demonstrate validation rejection with a deliberately invalid row.
    bad_row: dict[str, str] = {"schema": "hypofuse.nbest"}
    bad_path = target / "bad.jsonl"
    bad_path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")
    try:
        read_manifest(bad_path)
        print("ERROR: invalid row was not rejected", file=sys.stderr)
        return 1
    except SchemaError as exc:
        print(f"invalid row correctly rejected: {exc}")

    # 5. Normalize and score each pair.
    norm_cfg = NormalizationConfig(case_fold=True, strip_punctuation=True)
    ref_map: dict[str, str] = {
        r["utterance_id"]: r["text"] for r in ref_rows if r["schema"] == "hypofuse.reference"
    }
    header = f"{'utterance_id':<12} {'CER':>8} {'WER':>8}"
    print(header)
    print("-" * 32)
    for r in nbest_rows:
        uid = r["utterance_id"]
        if uid not in ref_map:
            continue
        hyp_text = r["hypotheses"][0]["text"] if r["hypotheses"] else ""
        ref_n, hyp_n = normalize_pair(ref_map[uid], hyp_text, norm_cfg)
        cer = character_error_rate(ref_n, hyp_n)
        wer = word_error_rate(ref_n.split(), hyp_n.split())
        print(f"{uid:<12} {cer:>8.4f} {wer:>8.4f}")

    return 0


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="ex01_")
    raise SystemExit(main(outdir))
