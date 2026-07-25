"""Deterministic answer checks (no LLM).

`check_grounding` verifies that every meaningful number in an answer actually came from
the query results (or the question) — a cheap, reliable guard against the model inventing
or miscalculating figures. It is a heuristic: it ignores small structural integers such as
list positions ("1.", "2.") and "top 3", and allows minor rounding.
"""

import re

# How close an answer number must be to a source number to count as supported.
_ABS_TOL = 0.02
_REL_TOL = 0.001

# Shorthand suffixes so "$99.5K" resolves to 99500 (≈ the exact figure, within tolerance).
_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}


def _numbers_in_text(text: str) -> list[float]:
    """Pull numeric values out of free text, handling $, thousands separators, and K/M/B."""
    nums = []
    # The K/M/B suffix must be attached to the digits ("$99.5K"), not a separate
    # letter ("2. B"), so there is no optional space before the suffix class.
    for match in re.findall(r"-?\$?\s?\d[\d,]*\.?\d*[kKmMbB]?", text or ""):
        cleaned = match.lower().replace("$", "").replace(",", "").replace(" ", "").strip()
        multiplier = 1.0
        if cleaned and cleaned[-1] in _SUFFIX:
            multiplier = _SUFFIX[cleaned[-1]]
            cleaned = cleaned[:-1]
        try:
            nums.append(float(cleaned) * multiplier)
        except ValueError:
            pass
    return nums


def _numbers_in_results(results) -> list[float]:
    """Collect every numeric value (including numeric strings like '2024') from rows."""
    nums = []
    for row in results or []:
        for value in row.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                nums.append(float(value))
            elif isinstance(value, str):
                try:
                    nums.append(float(value))
                except ValueError:
                    pass
    return nums


def _is_supported(number: float, sources: set) -> bool:
    """True if `number` matches any source value, allowing tiny/rounding differences.

    Also matches on absolute value: expenses are stored negative but are naturally
    reported as positive amounts (an expense of -$230.83 shown as "$230.83").
    """
    for source in sources:
        for a, b in ((number, source), (abs(number), abs(source))):
            if abs(a - b) <= _ABS_TOL:
                return True
            if abs(round(a) - b) <= 0.5:  # answer rounded to a whole number
                return True
            if b != 0 and abs(a - b) / abs(b) <= _REL_TOL:
                return True
    return False


def check_grounding(answer: str, results, question: str = "") -> dict:
    """Check that the numbers in `answer` are supported by `results` (or the question).

    Returns {grounded, unsupported, answer_numbers, allowed_numbers}. Small integers
    (< 100, e.g. list positions or "top 3") are skipped as structural, not financial.
    """
    allowed = set(_numbers_in_results(results)) | set(_numbers_in_text(question))
    answer_numbers = _numbers_in_text(answer)

    unsupported = []
    for number in answer_numbers:
        if abs(number) < 100 and float(number).is_integer():
            continue  # structural (ranks, counts), not a financial figure
        if not _is_supported(number, allowed):
            unsupported.append(number)

    return {
        "grounded": not unsupported,
        "unsupported": unsupported,
        "answer_numbers": answer_numbers,
        "allowed_numbers": sorted(allowed),
    }
