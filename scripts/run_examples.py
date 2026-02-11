"""Run all examples into a temporary directory.

Fully offline.  No network access, no third-party dependencies beyond
the project itself.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

EXAMPLES: tuple[str, ...] = (
    "ex01_validate_score",
    "ex02_rover_confusion",
    "ex03_rescore_calibrate",
)


def main() -> int:
    """Run each example and report exit codes."""
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="examples_"))
    base.mkdir(parents=True, exist_ok=True)

    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    failed = 0
    for name in EXAMPLES:
        script = examples_dir / name / "run.py"
        outdir = base / name
        outdir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, str(script), str(outdir)],
            capture_output=True,
            text=True,
        )
        status = "OK" if result.returncode == 0 else "FAIL"
        print(f"{name}: {status} (exit={result.returncode})")
        if result.returncode != 0:
            failed += 1
            if result.stderr:
                print(result.stderr, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
