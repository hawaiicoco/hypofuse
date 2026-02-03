"""Example ex02: ROVER fusion with confusion networks.

Fully offline, deterministic and synthetic.  No network access, no real
ASR output.  All hypotheses are generated in-process with seeded
generators.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from hypofuse.confusion import (
    build_confusion_network,
    confusion_to_json,
    minimal_cut_one_best,
)
from hypofuse.fixtures import FixtureConfig, generate_fixture
from hypofuse.fusion import FusionConfig, fuse, fusion_invariants
from hypofuse.multi_align import progressive_align


def main(outdir: str) -> int:
    """Run the ex02 example, writing confusion networks into *outdir*."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)

    cfg = FixtureConfig(n_utterances=4, n_best=3, seed=7)
    fixtures = generate_fixture(cfg)

    for fx in fixtures:
        hyps = list(fx.hypotheses)

        # Progressive alignment.
        grid = progressive_align(hyps)
        print(f"{fx.utterance_id}: grid {grid.width}x{grid.depth}")

        # Fuse with two policies.
        for policy in ("majority", "score_weighted"):
            fc = FusionConfig(policy=policy)
            result = fuse(grid, config=fc)
            all_tokens = [list(h) for h in hyps]
            ok = fusion_invariants(result, all_tokens)
            label = " ".join(result.tokens)
            print(f"  {policy}: {label} (invariants={ok})")
            if not ok:
                print(
                    f"ERROR: fusion invariants failed for {fx.utterance_id}",
                    file=sys.stderr,
                )
                return 1

        # Build confusion network.
        cn = build_confusion_network(grid)
        cn.validate()

        # Extract 1-best.
        one_best = minimal_cut_one_best(cn)
        print(f"  1-best: {' '.join(one_best)}")

        # Verify no novel tokens.
        input_tokens: set[str] = set()
        for h in hyps:
            input_tokens.update(h)
        novel = set(one_best) - input_tokens
        if novel:
            print(
                f"ERROR: fused tokens {novel} absent from inputs",
                file=sys.stderr,
            )
            return 1

        # Write confusion network JSON.
        cn_json = confusion_to_json(cn)
        (target / f"{fx.utterance_id}_cn.json").write_text(cn_json, encoding="utf-8")

    print(f"wrote {len(fixtures)} confusion network files")
    return 0


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="ex02_")
    raise SystemExit(main(outdir))
