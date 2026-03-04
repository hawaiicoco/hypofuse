"""Packaging smoke test: build wheel, install into clean venv, verify entry points.

Run from the project root::

    python scripts/check_package.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path


def _check(
    label: str,
    ok: bool,
    detail: str = "",
    *,
    fatal: bool = False,
) -> bool:
    """Print PASS/FAIL for a single check; exit immediately when *fatal*."""
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if detail and not ok:
        msg += f"  -- {detail}"
    print(msg)
    if not ok and fatal:
        raise SystemExit(1)
    return ok


def main() -> int:
    """Entry point for the packaging smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    with pyproject.open("rb") as fh:
        meta = tomllib.load(fh)
    expected_version: str = meta["project"]["version"]
    results: list[bool] = []

    with tempfile.TemporaryDirectory(prefix="hypofuse-pkg-check-") as tmpdir:
        tmp = Path(tmpdir)
        wheel_dir = tmp / "wheels"
        venv_dir = tmp / "venv"
        demo_out = tmp / "demo-out"
        wheel_dir.mkdir()
        demo_out.mkdir()

        # -- Build --------------------------------------------------------
        print("Building wheel ...")
        subprocess.check_call(
            [
                args.python,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheel_dir),
            ],
            cwd=root,
        )
        wheels = list(wheel_dir.glob("*.whl"))
        _check(
            "wheel produced",
            len(wheels) >= 1,
            f"found {len(wheels)}",
            fatal=True,
        )
        wheel_path = wheels[0]

        # -- Wheel contents -----------------------------------------------
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
        has_tests = any("tests/" in n for n in names)
        has_examples = any("examples/" in n for n in names)
        results.append(_check("no tests/ leaked into wheel", not has_tests))
        results.append(_check("no examples/ leaked into wheel", not has_examples))

        # -- Install into clean venv --------------------------------------
        print("Installing into clean venv ...")
        subprocess.check_call([args.python, "-m", "venv", str(venv_dir)])
        vpy = str(venv_dir / "bin" / "python")
        subprocess.check_call([vpy, "-m", "pip", "install", "--quiet", str(wheel_path)])

        # -- Import + version ---------------------------------------------
        out = subprocess.check_output(
            [vpy, "-c", "import hypofuse; print(hypofuse.__version__)"],
            text=True,
        ).strip()
        results.append(
            _check(
                "__version__ matches pyproject.toml",
                out == expected_version,
                f"got {out!r}, expected {expected_version!r}",
            )
        )

        # -- Console script: --version ------------------------------------
        hypofuse_bin = str(venv_dir / "bin" / "hypofuse")
        bin_exists = Path(hypofuse_bin).exists()
        results.append(_check("hypofuse console script installed", bin_exists))
        if bin_exists:
            ver_out = subprocess.check_output([hypofuse_bin, "--version"], text=True).strip()
            results.append(
                _check(
                    "hypofuse --version matches",
                    expected_version in ver_out,
                    f"got {ver_out!r}",
                )
            )

            # -- Console script: demo --out -------------------------------
            proc = subprocess.run(
                [hypofuse_bin, "demo", "--out", str(demo_out)],
                text=True,
                capture_output=True,
            )
            results.append(
                _check(
                    "hypofuse demo --out runs",
                    proc.returncode == 0,
                    proc.stderr.strip()[:200] if proc.returncode != 0 else "",
                )
            )
            results.append(
                _check(
                    "demo produced report.md",
                    (demo_out / "report.md").exists(),
                )
            )

        # -- Submodule imports --------------------------------------------
        for mod in ("hypofuse.fixtures", "hypofuse.cli"):
            proc = subprocess.run(
                [vpy, "-c", f"import {mod}"],
                text=True,
                capture_output=True,
            )
            results.append(
                _check(
                    f"import {mod}",
                    proc.returncode == 0,
                    proc.stderr.strip()[:200] if proc.returncode != 0 else "",
                )
            )

        shutil.rmtree(tmp, ignore_errors=True)

    if all(results):
        print("\nAll checks passed.")
        return 0
    failed = results.count(False)
    print(f"\n{failed} check(s) failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
