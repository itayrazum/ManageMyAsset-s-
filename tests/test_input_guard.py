"""Tests for the deterministic input guard (src/graph.py)."""

from src.graph import _validate_input


def test_empty_or_whitespace_is_flagged():
    assert _validate_input("")
    assert _validate_input("   ")
    assert _validate_input(None)


def test_overly_long_input_is_flagged():
    assert _validate_input("x" * 1001)


def test_normal_question_passes():
    assert _validate_input("What is my P&L in 2024?") == ""
