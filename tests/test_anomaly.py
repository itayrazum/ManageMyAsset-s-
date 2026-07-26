"""Tests for the anomaly-detection model (src/anomaly.py). No LLM calls."""

from src.anomaly import detect_anomalies, extract_period


def test_portfolio_flags_material_anomalies():
    anomalies = detect_anomalies()
    assert isinstance(anomalies, list) and anomalies
    assert {"category", "month", "value", "typical", "note"} <= set(anomalies[0])


def test_extract_period():
    assert extract_period("is anything weird in 2024?") == "2024"
    assert extract_period("anomalies in Q1 2025") == "2025-Q1"
    assert extract_period("anything unusual?") == ""


def test_period_scopes_to_that_window():
    # Scoping to 2024 must not report any 2025 month (e.g. the March 2025 consultancy spike).
    flagged = detect_anomalies(period="2024")
    assert flagged
    assert all(a["month"].startswith("2024-") for a in flagged)


def test_stable_scope_reports_nothing():
    # Building 17's categories are steady, so nothing material should be flagged.
    assert detect_anomalies(property="Building 17") == []


def test_deterministic():
    assert detect_anomalies() == detect_anomalies()
