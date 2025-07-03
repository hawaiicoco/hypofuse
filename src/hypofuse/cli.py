"""Command-line interface for hypofuse.

Exit-code policy:

* ``0`` -- success
* ``2`` -- user error (missing file, invalid manifest, bad flag value,
  :class:`~hypofuse.exceptions.HypofuseError` subclass, or ``ValueError``)

Unexpected exceptions may propagate with a traceback. ``demo`` is fully
offline; it generates a small synthetic dataset and runs the post-processing
pipeline end-to-end.
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
    p = sub.add_parser("rescore", help=COMMAND_HELP["rescore"])
    p.add_argument("--nbest", required=True, help="Path to n-best manifest.")
    lm = p.add_mutually_exclusive_group(required=True)
    lm.add_argument("--arpa", help="Path to ARPA language model file.")
    lm.add_argument("--corpus", help="Path to text corpus (one sentence per line).")
    p.add_argument("--order", type=int, default=3, help="N-gram order for corpus training.")
    p.add_argument("--lm-weight", type=float, default=0.5, help="LM weight for fusion.")
    p.add_argument("--acoustic-weight", type=float, default=1.0, help="Acoustic weight for fusion.")
    p.add_argument("--top", type=int, default=1, help="Number of top hypotheses to show.")
    p.add_argument("--json", action="store_true", help="Emit JSON output.")


def _add_calibrate_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("calibrate", help=COMMAND_HELP["calibrate"])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scores", help="Comma-separated scores.")
    src.add_argument("--scores-file", help="Path to file with one score per line.")
    p.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter.")
    p.add_argument(
        "--method",
        default="temperature",
        choices=["temperature", "piecewise"],
        help="Calibration method.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON output.")


def _add_analyze_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("analyze", help=COMMAND_HELP["analyze"])
    p.add_argument("--nbest", required=True, help="Path to n-best manifest.")
    p.add_argument("--reference", required=True, help="Path to reference manifest.")
    p.add_argument(
        "--slice-by",
        default="speaker_group",
        choices=["speaker_group", "intent_domain", "noise_db", "duration_s"],
        help="Field to slice by.",
    )
    p.add_argument("--buckets", type=int, default=3, help="Number of duration buckets.")
    p.add_argument("--json", action="store_true", help="Emit JSON output.")


def _add_report_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("report", help=COMMAND_HELP["report"])
    p.add_argument("--nbest", required=True, help="Path to n-best manifest.")
    p.add_argument("--reference", required=True, help="Path to reference manifest.")
    p.add_argument(
        "--format",
        default="markdown",
        choices=["markdown", "jsonl"],
        help="Output format.",
    )
    p.add_argument("--out", type=str, default=None, help="Output file path (default: stdout).")
    p.add_argument("--compare", type=str, default=None, help="System ID for comparison.")


def _add_demo_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("demo", help=COMMAND_HELP["demo"])
    p.add_argument("--out", type=str, default=".", help="Output directory.")
    p.add_argument("--utterances", type=int, default=5, help="Number of utterances.")
    p.add_argument("--n-best", type=int, default=3, help="N-best list size.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")


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
    from hypofuse.ngram import NgramLM
    from hypofuse.rescore import ScoredHypothesis, rescore_nbest

    nbest_path = Path(args.nbest)
    if not nbest_path.exists():
        print(f"hypofuse rescore: file not found: {nbest_path}", file=sys.stderr)
        return 2
    if args.arpa is not None:
        arpa_path = Path(args.arpa)
        if not arpa_path.exists():
            print(f"hypofuse rescore: file not found: {arpa_path}", file=sys.stderr)
            return 2
        lm = NgramLM.from_arpa(arpa_path.read_text(encoding="utf-8"))
    else:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(f"hypofuse rescore: file not found: {corpus_path}", file=sys.stderr)
            return 2
        sentences = [
            line.split()
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not sentences:
            print("hypofuse rescore: empty corpus", file=sys.stderr)
            return 2
        lm = NgramLM.train(sentences, order=args.order)
    rows = read_manifest(nbest_path)
    all_results: list[dict[str, object]] = []
    for r in rows:
        if r["schema"] != "hypofuse.nbest":
            continue
        acoustic = float(r.get("acoustic_log10", 0.0))
        hyps = [
            ScoredHypothesis.from_tokens(h["tokens"], acoustic_log10=acoustic)
            for h in r.get("hypotheses", [])
            if "tokens" in h
        ]
        if not hyps:
            continue
        ranked = rescore_nbest(hyps, lm, args.lm_weight, args.acoustic_weight)
        top_k = ranked[: args.top]
        all_results.append(
            {
                "utterance_id": r["utterance_id"],
                "hypotheses": [
                    {
                        "text": h.text,
                        "acoustic_log10": h.acoustic_log10,
                        "lm_log10": h.lm_log10,
                    }
                    for h in top_k
                ],
            }
        )
    if not all_results:
        print("hypofuse rescore: no n-best rows found", file=sys.stderr)
        return 2
    print(json.dumps(all_results))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    raw = args.scores
    if raw is None:
        p = Path(args.scores_file)
        if not p.exists():
            print(f"hypofuse calibrate: file not found: {p}", file=sys.stderr)
            return 2
        raw = p.read_text(encoding="utf-8").strip().replace("\n", ",")
    if not raw or not raw.strip():
        print("hypofuse calibrate: empty scores list", file=sys.stderr)
        return 2
    try:
        scores = [float(s) for s in raw.split(",") if s.strip()]
    except ValueError as exc:
        print(f"hypofuse calibrate: non-numeric input: {exc}", file=sys.stderr)
        return 2
    if not scores:
        print("hypofuse calibrate: empty scores list", file=sys.stderr)
        return 2
    if args.method == "temperature":
        out = temperature_scale(scores, temperature=args.temperature)
    else:
        from hypofuse.confidence import piecewise_calibrate

        out = piecewise_calibrate(scores, [0.0], [1.0], [0.0])
    print(json.dumps(out))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from hypofuse.analysis import (
        UtteranceScore,
        slice_by_duration,
        slice_by_field,
        slice_metrics,
    )

    nbest_path = Path(args.nbest)
    ref_path = Path(args.reference)
    for label, p in [("nbest", nbest_path), ("reference", ref_path)]:
        if not p.exists():
            print(f"hypofuse analyze: {label} file not found: {p}", file=sys.stderr)
            return 2
    nbest_rows = read_manifest(nbest_path)
    ref_rows = read_manifest(ref_path)
    ref_map = {r["utterance_id"]: r for r in ref_rows if r["schema"] == "hypofuse.reference"}
    items: list[UtteranceScore] = []
    for r in nbest_rows:
        if r["schema"] != "hypofuse.nbest":
            continue
        uid = r["utterance_id"]
        if uid not in ref_map:
            continue
        rr = ref_map[uid]
        hyp_tokens = tuple(r["hypotheses"][0]["tokens"]) if r.get("hypotheses") else ()
        ref_tokens = tuple(rr["text"].split())
        items.append(
            UtteranceScore(
                utterance_id=uid,
                reference=ref_tokens,
                hypothesis=hyp_tokens,
                speaker_group=rr.get("speaker_group", ""),
                intent_domain=rr.get("intent_domain", ""),
                noise_db=float(rr.get("noise_db", 0.0)),
                duration_s=float(rr.get("duration_s", 0.0)),
            )
        )
    if not items:
        print("hypofuse analyze: no matched utterances", file=sys.stderr)
        return 2
    if args.slice_by == "duration_s":
        boundaries = tuple(round(i * 12.0 / args.buckets, 1) for i in range(1, args.buckets))
        slices = slice_by_duration(items, boundaries)
    else:
        field_map = {
            "speaker_group": "speaker_group",
            "intent_domain": "intent_domain",
            "noise_db": "noise",
        }
        slices = slice_by_field(items, field_map[args.slice_by])
    metrics = slice_metrics(slices)
    if args.json:
        print(
            json.dumps(
                [{"key": m.key, "count": m.count, "cer": m.cer, "wer": m.wer} for m in metrics]
            )
        )
    else:
        print(f"{'Slice':<20} {'Count':>6} {'CER':>8} {'WER':>8}")
        for m in metrics:
            print(f"{m.key:<20} {m.count:>6} {m.cer:>8.4f} {m.wer:>8.4f}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from hypofuse.analysis import (
        UtteranceScore,
        report_to_jsonl,
        report_to_markdown,
        slice_by_field,
        slice_metrics,
    )

    nbest_path = Path(args.nbest)
    ref_path = Path(args.reference)
    for label, p in [("nbest", nbest_path), ("reference", ref_path)]:
        if not p.exists():
            print(f"hypofuse report: {label} file not found: {p}", file=sys.stderr)
            return 2
    nbest_rows = read_manifest(nbest_path)
    ref_rows = read_manifest(ref_path)
    ref_map = {r["utterance_id"]: r for r in ref_rows if r["schema"] == "hypofuse.reference"}
    items: list[UtteranceScore] = []
    for r in nbest_rows:
        if r["schema"] != "hypofuse.nbest":
            continue
        uid = r["utterance_id"]
        if uid not in ref_map:
            continue
        rr = ref_map[uid]
        hyp_tokens = tuple(r["hypotheses"][0]["tokens"]) if r.get("hypotheses") else ()
        ref_tokens = tuple(rr["text"].split())
        items.append(
            UtteranceScore(
                utterance_id=uid,
                reference=ref_tokens,
                hypothesis=hyp_tokens,
                speaker_group=rr.get("speaker_group", ""),
                intent_domain=rr.get("intent_domain", ""),
                noise_db=float(rr.get("noise_db", 0.0)),
                duration_s=float(rr.get("duration_s", 0.0)),
            )
        )
    if not items:
        print("hypofuse report: no matched utterances", file=sys.stderr)
        return 2
    slices = slice_by_field(items, "speaker_group")
    metrics = slice_metrics(slices)
    comparisons: list[object] = []
    if args.format == "markdown":
        text = report_to_markdown(metrics, comparisons)
    else:
        text = report_to_jsonl(metrics, comparisons)
    if args.out is not None:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote report to {args.out}")
    else:
        print(text)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from hypofuse.alignment import character_error_rate, word_error_rate

    cfg = FixtureConfig(n_utterances=args.utterances, n_best=args.n_best, seed=args.seed)
    fx = generate_fixture(cfg)
    rows = as_manifest_dicts(fx)
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    nbest_rows = [r for r in rows if r["schema"] == "hypofuse.nbest"]
    ref_rows = [r for r in rows if r["schema"] == "hypofuse.reference"]
    write_manifest(target / "nbest.jsonl", nbest_rows)
    write_manifest(target / "reference.jsonl", ref_rows)
    refs = {r["utterance_id"]: r["text"] for r in ref_rows}
    cer_vals: list[float] = []
    wer_vals: list[float] = []
    for r in nbest_rows:
        uid = r["utterance_id"]
        if uid not in refs:
            continue
        hyp_text = r["hypotheses"][0]["text"] if r["hypotheses"] else ""
        cer_vals.append(character_error_rate(refs[uid], hyp_text))
        wer_vals.append(word_error_rate(refs[uid].split(), hyp_text.split()))
    avg_cer = sum(cer_vals) / len(cer_vals) if cer_vals else 0.0
    avg_wer = sum(wer_vals) / len(wer_vals) if wer_vals else 0.0
    by_uid: dict[str, list[list[str]]] = {}
    for r in nbest_rows:
        by_uid.setdefault(r["utterance_id"], []).extend(
            [h["tokens"] for h in r.get("hypotheses", []) if "tokens" in h]
        )
    fused_count = 0
    for hyps in by_uid.values():
        if not hyps:
            continue
        grid = progressive_align(hyps)
        fuse(grid, config=FusionConfig())
        fused_count += 1
    report_lines = [
        "# Demo Report",
        "",
        f"Utterances: {args.utterances}",
        f"N-best: {args.n_best}",
        f"Seed: {args.seed}",
        "",
        f"Mean CER: {avg_cer:.4f}",
        f"Mean WER: {avg_wer:.4f}",
        "",
        f"Fused {fused_count} utterances.",
    ]
    (target / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"demo: wrote manifests and report to {target}")
    print(f"  CER={avg_cer:.4f} WER={avg_wer:.4f} fused={fused_count}")
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
    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return handler(args)
    except (HypofuseError, ValueError) as exc:
        print(f"hypofuse {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
