"""Build a wheel into a temporary directory and install it into a clean venv
to confirm packaging is healthy. Run inside the project root.

Usage: python scripts/check_package.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="hypofuse-pkg-check-") as tmpdir:
        wheel_dir = Path(tmpdir) / "wheels"
        venv_dir = Path(tmpdir) / "venv"
        wheel_dir.mkdir()
        subprocess.check_call(
            [args.python, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(wheel_dir)],
            cwd=root,
        )
        wheels = list(wheel_dir.glob("*.whl"))
        if not wheels:
            print("no wheel produced", file=sys.stderr)
            return 2
        subprocess.check_call([args.python, "-m", "venv", str(venv_dir)])
        site = venv_dir / "lib" / "python3.11" / "site-packages"
        if not site.exists():
            # Fallback for non-Debian layouts.
            site = next((venv_dir / "lib").glob("python*/site-packages"))
        for whl in wheels:
            subprocess.check_call(
                [str(venv_dir / "bin" / "python"), "-m", "pip", "install", "--quiet", str(whl)],
                env={**os.environ, "PIP_NO_INDEX": "1"},
            )
        # Confirm the installed package imports.
        cmd = [
            str(venv_dir / "bin" / "python"),
            "-c",
            "import hypofuse; print(hypofuse.__version__)",
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        print(f"installed wheel reports: {out}")
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
