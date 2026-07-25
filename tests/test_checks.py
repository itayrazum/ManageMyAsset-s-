"""Tests for the deterministic grounding check (src/checks.py)."""

from src.checks import check_grounding


def test_faithful_answer_is_grounded():
    data = [{"total": 361810.32}]
    assert check_grounding("Profit was $361,810.32.", data)["grounded"]


def test_fabricated_number_is_flagged():
    data = [{"total": 361810.32}]
    result = check_grounding("Profit was $500,000.00.", data)
    assert not result["grounded"]
    assert 500000.0 in result["unsupported"]


def test_shorthand_resolves():
    # "$99.5K" == 99500, within tolerance of the exact 99,501.25
    assert check_grounding("A difference of about $99.5K.", [{"d": 99501.25}])["grounded"]


def test_expenses_reported_as_positive_are_grounded():
    # Expenses are stored negative but naturally reported as a positive amount.
    assert check_grounding("Expenses were $230,313.83.", [{"e": -230313.83}])["grounded"]


def test_structural_integers_are_ignored():
    # List positions ("1.", "2.", "3.") and "Top 3" are not financial figures.
    data = [{"x": 361810.32}]
    assert check_grounding("Top 3: 1. A  2. B  3. C ($361,810.32)", data)["grounded"]


def test_billion_suffix_not_triggered_by_list_label():
    # "2. B" must not be read as 2 billion.
    assert check_grounding("2. B ($361,810.32)", [{"x": 361810.32}])["grounded"]
