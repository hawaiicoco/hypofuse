"""Scale test: 2000 utterances with rate tolerance check."""

from __future__ import annotations

import time

import pytest

from hypofuse.alignment import DEL, INS, SUB, edit_alignment
from hypofuse.fixtures import FixtureConfig, generate_fixture


@pytest.mark.slow
def test_scale_2000_utterances() -> None:
    """Generate 2000 utterances and verify measured rates.

    Tolerance is 0.01 absolute (~3 standard errors for n~11000
    tokens at the configured rates).
    """
    cfg = FixtureConfig(
        n_utterances=2000,
        n_best=1,
        seed=42,
        substitution_rate=0.05,
        insertion_rate=0.02,
        deletion_rate=0.02,
    )
    t0 = time.monotonic()
    fx = generate_fixture(cfg)
    elapsed = time.monotonic() - t0
    assert elapsed < 30.0, f"generation took {elapsed:.1f}s"
    total_ref = 0
    subs = 0
    inss = 0
    dels = 0
    for u in fx:
        ops = edit_alignment(u.reference, u.hypotheses[0]).ops
        total_ref += len(u.reference)
        for op in ops:
            if op.op == SUB:
                subs += 1
            elif op.op == INS:
                inss += 1
            elif op.op == DEL:
                dels += 1
    tol = 0.01
    measured_sub = subs / total_ref
    measured_ins = inss / total_ref
    measured_del = dels / total_ref
    assert abs(measured_sub - 0.05) < tol, f"sub rate {measured_sub:.4f}"
    assert abs(measured_ins - 0.02) < tol, f"ins rate {measured_ins:.4f}"
    assert abs(measured_del - 0.02) < tol, f"del rate {measured_del:.4f}"
