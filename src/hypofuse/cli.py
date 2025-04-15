"""Command-line interface for hypofuse.

Every subcommand exits with a non-zero code on invalid input. ``demo`` is
fully offline; it generates a small synthetic dataset and runs the
post-processing pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from hypofuse import __version__
from hypofuse.confidence import temperature_scale
from hypofuse.fixtures import FixtureConfig, as_manifest_dicts, generate_fixture
from hypofuse.fusion import FusionConfig, fuse
from hypofuse.manifests.jsonl import read_manifest, write_manifest
from hypofuse.multi_align import progressive_align
from hypofuse.normalize import NormalizationConfig, normalize_pair


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypofuse",
        description="Speech recognition post-processing toolkit (offline, deterministic).",
    )
    parser.add_argument("--version", action="version", version=f"hypofuse {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="Validate a JSONL manifest file.")
    sub.add_parser("normalize", help="Normalize reference + hypothesis.")
    sub.add_parser("score", help="Compute CER / WER over a manifest.")
    sub.add_parser("align", help="Run multi-hypothesis alignment and print the grid.")
    sub.add_parser("fuse", help="Fuse an n-best list with the chosen policy.")
    sub.add_parser("rescore", help="Re-rank an n-best list with an LM.")
    sub.add_parser("calibrate", help="Apply temperature scaling to per-token scores.")
    sub.add_parser("analyze", help="Slice-based error analysis.")
    sub.add_parser("report", help="Render Markdown / JSONL reports.")
    sub.add_parser("demo", help="Run a fully offline end-to-end demo on synthetic data.")
    return parser


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        rows = read_manifest(args.path)
    except Exception as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2
    print(f"validated {len(rows)} records")
    return 0


def _cmd_normalize(args: argparse.Namespace) -> int:
    cfg = NormalizationConfig()
    ref, hyp = normalize_pair(args.reference, args.hypothesis, cfg)
    print(json.dumps({"reference": ref, "hypothesis": hyp}))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from hypofuse.alignment import character_error_rate, word_error_rate

    rows = read_manifest(args.path)
    cer_vals = []
    wer_vals = []
    refs = {r["utterance_id"]: r["text"] for r in rows if r["schema"] == "hypofuse.reference"}
    for r in rows:
        if r["schema"] != "hypofuse.nbest":
            continue
        if r["utterance_id"] not in refs:
            continue
        ref_text = refs[r["utterance_id"]]
        hyp_text = r["hypotheses"][0]["text"] if r["hypotheses"] else ""
        cer_vals.append(character_error_rate(ref_text, hyp_text))
        wer_vals.append(word_error_rate(ref_text.split(), hyp_text.split()))
    if not cer_vals:
        print("no scored pairs", file=sys.stderr)
        return 2
    avg_cer = sum(cer_vals) / len(cer_vals)
    avg_wer = sum(wer_vals) / len(wer_vals)
    print(json.dumps({"cer": avg_cer, "wer": avg_wer, "n": len(cer_vals)}))
    return 0


def _cmd_align(args: argparse.Namespace) -> int:
    rows = read_manifest(args.path)
    by_uid: dict[str, list[list[str]]] = {}
    for r in rows:
        if r["schema"] != "hypofuse.nbest":
            continue
        by_uid.setdefault(r["utterance_id"], []).append(
            [h["tokens"] for h in r.get("hypotheses", []) if "tokens" in h]
        )
    for uid, groups in by_uid.items():
        grid = progressive_align(groups)
        print(json.dumps({"utterance_id": uid, "width": grid.width, "depth": grid.depth}))
    return 0


def _cmd_fuse(args: argparse.Namespace) -> int:
    rows = read_manifest(args.path)
    cfg = FusionConfig(policy=args.policy)
    by_uid: dict[str, list[list[str]]] = {}
    for r in rows:
        if r["schema"] != "hypofuse.nbest":
            continue
        by_uid.setdefault(r["utterance_id"], []).append(
            [h["tokens"] for h in r.get("hypotheses", []) if "tokens" in h]
        )
    for uid, hyps in by_uid.items():
        if not hyps:
            continue
        grid = progressive_align(hyps)
        result = fuse(grid, config=cfg)
        print(json.dumps({"utterance_id": uid, "tokens": list(result.tokens)}))
    return 0


def _cmd_rescore(args: argparse.Namespace) -> int:
    print("rescore: not yet wired in stub", file=sys.stderr)
    return 1


def _cmd_calibrate(args: argparse.Namespace) -> int:
    scores = [float(s) for s in args.scores.split(",")]
    out = temperature_scale(scores, temperature=args.temperature)
    print(json.dumps(out))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    print("analyze: not yet wired in stub", file=sys.stderr)
    return 1


def _cmd_report(args: argparse.Namespace) -> int:
    print("report: not yet wired in stub", file=sys.stderr)
    return 1


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = FixtureConfig(n_utterances=5, n_best=3, seed=0)
    fx = generate_fixture(cfg)
    rows = as_manifest_dicts(fx)
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    write_manifest(target / "manifest.jsonl", rows)
    print(f"wrote {target / 'manifest.jsonl'} with {len(rows)} rows")
    return 0


_HANDLERS = {
    "validate": _cmd_validate,
    "normalize": _cmd_normalize,
    "score": _cmd_score,
    "align": _cmd_align,
    "fuse": _cmd_fuse,
    "rescore": _cmd_rescore,
    "calibrate": _cmd_calibrate,
    "analyze": _cmd_analyze,
    "report": _cmd_report,
    "demo": _cmd_demo,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    # Inject shared args based on command.
    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        return 2
    # Build per-subcommand args.
    return _dispatch(args, handler)


def _dispatch(args: argparse.Namespace, handler) -> int:
    if args.command == "validate":
        args.path = _ask_path(args)
    elif args.command == "normalize":
        args.reference = input("reference> ")
        args.hypothesis = input("hypothesis> ")
    elif args.command == "score" or args.command == "align":
        args.path = _ask_path(args)
    elif args.command == "fuse":
        args.policy = input("policy [majority|score_weighted|lm_weighted]> ") or "majority"
        args.path = _ask_path(args)
    elif args.command == "calibrate":
        args.scores = input("scores (comma-separated)> ")
        try:
            args.temperature = float(input("temperature> ") or "1.0")
        except ValueError:
            args.temperature = 1.0
    elif args.command == "demo":
        args.out = input("output dir> ") or "."
    return handler(args)


def _ask_path(args: argparse.Namespace) -> str:
    return input("manifest path> ")


if __name__ == "__main__":
    raise SystemExit(main())
