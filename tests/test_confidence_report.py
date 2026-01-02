"""Calibration report and golden JSON serialization."""

from __future__ import annotations

import json
import math

import pytest

from hypofuse.confidence import calibration_report, calibration_report_to_json


def test_report_structure() -> None:
    report = calibration_report([0.5, 0.5], [1.0, 0.0], bins=2)
    assert "ece" in report
    assert "brier_score" in report
    assert "log_loss" in report
    assert "bins" in report
    assert "binning" in report
    assert report["binning"] == "uniform"
    assert len(report["bins"]) == 2


def test_report_values() -> None:
    # ECE: all in bin 1, avg_conf=0.5, avg_acc=0.5, ECE=0
    # Brier: mean(0.25, 0.25) = 0.25
    # Log loss: mean(ln(2), ln(2)) = ln(2)
    report = calibration_report([0.5, 0.5], [1.0, 0.0], bins=2)
    assert report["ece"] == pytest.approx(0.0)
    assert report["brier_score"] == pytest.approx(0.25)
    assert report["log_loss"] == pytest.approx(math.log(2))


def test_report_to_json_round_trip() -> None:
    report = calibration_report([0.5, 0.5], [1.0, 0.0], bins=2)
    text = calibration_report_to_json(report)
    parsed = json.loads(text)
    assert parsed["ece"] == pytest.approx(0.0)
    assert parsed["brier_score"] == pytest.approx(0.25)


def test_golden_json_12_point() -> None:
    # 12 points: all probs=0.5, labels alternate 1/0, bins=2
    # All go to bin 1 (int(0.5*2)=1).
    # Bin 0: empty. Bin 1: 12 points, avg_conf=0.5, avg_acc=0.5.
    # ECE=0.0, Brier=0.25, Log_loss=ln(2)
    probs = [0.5] * 12
    labels = [1.0, 0.0] * 6
    report = calibration_report(probs, labels, bins=2)
    text = calibration_report_to_json(report)
    expected = (
        "{\n"
        '  "binning": "uniform",\n'
        '  "bins": [\n'
        "    {\n"
        '      "avg_accuracy": 0.0,\n'
        '      "avg_confidence": 0.0,\n'
        '      "count": 0,\n'
        '      "lower": 0.0,\n'
        '      "upper": 0.5\n'
        "    },\n"
        "    {\n"
        '      "avg_accuracy": 0.5,\n'
        '      "avg_confidence": 0.5,\n'
        '      "count": 12,\n'
        '      "lower": 0.5,\n'
        '      "upper": 1.0\n'
        "    }\n"
        "  ],\n"
        '  "brier_score": 0.25,\n'
        '  "ece": 0.0,\n'
        '  "log_loss": 0.6931471805599453\n'
        "}"
    )
    assert text == expected
