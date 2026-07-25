"""Tests for the SQL safety guard and executor (src/agents/sql_analyst.py).

These are deterministic — they run DuckDB locally and make no LLM/API calls.
"""

from src.agents.sql_analyst import _execute_sql, _is_safe


def test_select_is_allowed():
    assert _is_safe("SELECT * FROM ledger")
    assert _is_safe("WITH x AS (SELECT 1) SELECT * FROM x")


def test_mutations_are_blocked():
    for query in ["DROP TABLE ledger", "DELETE FROM ledger",
                  "UPDATE ledger SET profit = 0", "INSERT INTO ledger VALUES (1)"]:
        assert not _is_safe(query)


def test_stacked_statements_are_blocked():
    assert not _is_safe("SELECT 1; DROP TABLE ledger")


def test_filesystem_writes_are_blocked():
    assert not _is_safe("COPY ledger TO 'out.csv'")


def test_execute_returns_rows_for_valid_query():
    result = _execute_sql("SELECT COUNT(*) AS n FROM ledger")
    assert "data" in result
    assert result["data"][0]["n"] > 0


def test_execute_returns_error_for_bad_column():
    result = _execute_sql("SELECT nope FROM ledger")
    assert "error" in result


def test_execute_rejects_unsafe_query():
    result = _execute_sql("DELETE FROM ledger")
    assert result.get("error")
