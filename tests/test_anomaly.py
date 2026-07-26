"""Tests for the anomaly-detection model (src/anomaly.py). No LLM calls."""

from src.anomaly import detect_anomalies


def test_portfolio_flags_material_anomalies():
    anomalies = detect_anomalies()
    assert isinstance(anomalies, list) and anomalies
    assert {"category", "month", "value", "typical", "note"} <= set(anomalies[0])


def test_stable_scope_reports_nothing():
    # Building 17's categories are steady, so nothing material should be flagged.
    assert detect_anomalies(property="Building 17") == []


def test_deterministic():
    assert detect_anomalies() == detect_anomalies()
