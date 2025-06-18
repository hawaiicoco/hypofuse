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
from hypofuse.exceptions import HypofuseError
from hypofuse.fixtures import FixtureConfig, as_manifest_dicts, generate_fixture
from hypofuse.fusion import FusionConfig, fuse
from hypofuse.manifests.jsonl import read_manifest, write_manifest
from hypofuse.multi_align import progressive_align
from hypofuse.normalize import NormalizationConfig, normalize_pair

COMMANDS: tuple[str, ...] = (
    "validate",
    "normalize",
    "score",
    "align",
    "fuse",
    "rescore",
    "calibrate",
    "analyze",
    "report",
    "demo",
)

COMMAND_HELP: dict[str, str] = {
    "validate": "Validate a JSONL manifest file.",
    "normalize": "Normalize reference + hypothesis.",
    "score": "Compute CER / WER over a manifest.",
    "align": "Run multi-hypothesis alignment and print the grid.",
    "fuse": "Fuse an n-best list with the chosen policy.",
    "rescore": "Re-rank an n-best list with an LM.",
    "calibrate": "Apply temperature scaling to per-token scores.",
    "analyze": "Slice-based error analysis.",
    "report": "Render Markdown / JSONL reports.",
    "demo": "Run a fully offline end-to-end demo on synthetic data.",
}


def _add_validate_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("validate", help=COMMAND_HELP["validate"])
    p.add_argument("path", type=str, help="Path to the JSONL manifest.")
    p.add_argument("--schema", type=str, default=None, help="Expected schema name.")
    p.add_argument("--json", action="store_true", help="Emit JSON output.")


def _add_normalize_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("normalize", help=COMMAND_HELP["normalize"])
    p.add_argument("--reference", required=True, help="Reference text.")
    p.add_argument("--hypothesis", required=True, help="Hypothesis text.")
    p.add_argument("--language", default="en", choices=["en", "zh"], help="Language hint.")
    p.add_argument("--keep-case", action="store_true", help="Disable case folding.")
    p.add_argument("--keep-punct", action="store_true", help="Keep punctuation.")
    p.add_argument("--json", action="store_true", help="Emit JSON (default).")


def _add_score_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("score", help=COMMAND_HELP["score"])
    p.add_argument("--nbest", required=True, help="Path to n-best manifest.")
    p.add_argument("--reference", required=True, help="Path to reference manifest.")
    p.add_argument(
        "--metric",
        default="both",
        choices=["cer", "wer", "both"],
        help="Metric to compute.",
    )
    norm = p.add_mutually_exclusive_group()
    norm.add_argument(
        "--normalize", action="store_true", dest="do_normalize", help="Normalize (default)."
    )
    norm.add_argument(
        "--no-normalize", action="store_false", dest="do_normalize", help="Skip normalization."
    )
    p.set_defaults(do_normalize=True)
    p.add_argument("--json", action="store_true", help="Emit JSON output.")


def _add_align_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("align", help=COMMAND_HELP["align"])
    p.add_argument("--nbest", required=True, help="Path to n-best manifest.")
    p.add_argument("--json", action="store_true", help="Emit JSON output.")


def _add_fuse_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("fuse", help=COMMAND_HELP["fuse"])
    p.add_argument("--nbest", required=True, help="Path to n-best manifest.")
    p.add_argument(
        "--policy",
        default="majority",
        choices=["majority", "score_weighted", "lm_weighted"],
        help="Fusion policy.",
    )
    p.add_argument(
        "--tie-break",
        default="lexicographic",
        choices=["lexicographic", "first"],
        help="Tie-breaking strategy.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON output.")


def _add_rescore_args(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("rescore", help=COMMAND_HELP["rescore"])


def _add_calibrate_args(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("calibrate", help=COMMAND_HELP["calibrate"])


def _add_analyze_args(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("analyze", help=COMMAND_HELP["analyze"])


def _add_report_args(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("report", help=COMMAND_HELP["report"])


def _add_demo_args(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("demo", help=COMMAND_HELP["demo"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypofuse",
        description="Speech recognition post-processing toolkit (offline, deterministic).",
    )
    parser.add_argument("--version", action="version", version=f"hypofuse {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_validate_args(sub)
    _add_normalize_args(sub)
    _add_score_args(sub)
    _add_align_args(sub)
    _add_fuse_args(sub)
    _add_rescore_args(sub)
    _add_calibrate_args(sub)
    _add_analyze_args(sub)
    _add_report_args(sub)
    _add_demo_args(sub)
    return parser


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"hypofuse validate: file not found: {path}", file=sys.stderr)
        return 2
    try:
        rows = read_manifest(path)
    except (HypofuseError, ValueError) as exc:
        print(f"hypofuse validate: {exc}", file=sys.stderr)
        return 2
    if args.schema is not None:
        for idx, row in enumerate(rows):
            if row.get("schema") != args.schema:
                msg = f"row {idx + 1}: expected {args.schema!r}, got {row.get('schema')!r}"
                print(f"hypofuse validate: {msg}", file=sys.stderr)
                return 2
    if args.json:
        print(json.dumps({"valid": True, "count": len(rows)}))
    else:
        print(f"validated {len(rows)} records")
    return 0


def _cmd_normalize(args: argparse.Namespace) -> int:
    cfg = NormalizationConfig(
        case_fold=not args.keep_case,
        strip_punctuation=not args.keep_punct,
        language_hint=args.language,
    )
    try:
        ref, hyp = normalize_pair(args.reference, args.hypothesis, cfg)
    except (HypofuseError, ValueError) as exc:
        print(f"hypofuse normalize: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"reference": ref, "hypothesis": hyp}))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from hypofuse.alignment import character_error_rate, word_error_rate

    nbest_path = Path(args.nbest)
    ref_path = Path(args.reference)
    for label, p in [("nbest", nbest_path), ("reference", ref_path)]:
        if not p.exists():
            print(f"hypofuse score: {label} file not found: {p}", file=sys.stderr)
            return 2
    nbest_rows = read_manifest(nbest_path)
    ref_rows = read_manifest(ref_path)
    refs = {r["utterance_id"]: r["text"] for r in ref_rows if r["schema"] == "hypofuse.reference"}
    cer_vals: list[float] = []
    wer_vals: list[float] = []
    for r in nbest_rows:
        if r["schema"] != "hypofuse.nbest":
            continue
        uid = r["utterance_id"]
        if uid not in refs:
            continue
        ref_text = refs[uid]
        hyp_text = r["hypotheses"][0]["text"] if r["hypotheses"] else ""
        if args.do_normalize:
            from hypofuse.normalize import normalize

            ref_text = normalize(ref_text)
            hyp_text = normalize(hyp_text)
        cer_vals.append(character_error_rate(ref_text, hyp_text))
        wer_vals.append(word_error_rate(ref_text.split(), hyp_text.split()))
    if not cer_vals:
        print("hypofuse score: no matched utterance ids", file=sys.stderr)
        return 2
    avg_cer = sum(cer_vals) / len(cer_vals)
    avg_wer = sum(wer_vals) / len(wer_vals)
    result: dict[str, float | int] = {"n": len(cer_vals)}
    if args.metric in ("cer", "both"):
        result["cer"] = avg_cer
    if args.metric in ("wer", "both"):
        result["wer"] = avg_wer
    print(json.dumps(result))
    return 0


def _cmd_align(args: argparse.Namespace) -> int:
    nbest_path = Path(args.nbest)
    if not nbest_path.exists():
        print(f"hypofuse align: file not found: {nbest_path}", file=sys.stderr)
        return 2
    rows = read_manifest(nbest_path)
    by_uid: dict[str, list[list[str]]] = {}
    for r in rows:
        if r["schema"] != "hypofuse.nbest":
            continue
        # One manifest row carries the whole n-best list, so extend (not append).
        by_uid.setdefault(r["utterance_id"], []).extend(
            [h["tokens"] for h in r.get("hypotheses", []) if "tokens" in h]
        )
    if not by_uid:
        print("hypofuse align: no n-best rows found", file=sys.stderr)
        return 2
    for uid, groups in by_uid.items():
        grid = progressive_align(groups)
        print(json.dumps({"utterance_id": uid, "width": grid.width, "depth": grid.depth}))
    return 0


def _cmd_fuse(args: argparse.Namespace) -> int:
    nbest_path = Path(args.nbest)
    if not nbest_path.exists():
        print(f"hypofuse fuse: file not found: {nbest_path}", file=sys.stderr)
        return 2
    rows = read_manifest(nbest_path)
    cfg = FusionConfig(policy=args.policy, tie_break=args.tie_break)
    by_uid: dict[str, list[list[str]]] = {}
    for r in rows:
        if r["schema"] != "hypofuse.nbest":
            continue
        by_uid.setdefault(r["utterance_id"], []).extend(
            [h["tokens"] for h in r.get("hypotheses", []) if "tokens" in h]
        )
    if not by_uid:
        print("hypofuse fuse: no n-best rows found", file=sys.stderr)
        return 2
    for uid, hyps in by_uid.items():
        if not hyps:
            continue
        grid = progressive_align(hyps)
        result = fuse(grid, config=cfg)
        print(
            json.dumps(
                {
                    "utterance_id": uid,
                    "tokens": list(result.tokens),
                    "confidences": list(result.confidences),
                }
            )
        )
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
    if args.command == "calibrate":
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
