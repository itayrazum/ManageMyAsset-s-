"""Tests for the structured answer cache key (src/cache.py)."""

from src import cache

_REV_2025 = {"metric": "revenue", "property": "", "tenant": "", "timeframe": "2025"}
_EMPTY = {"metric": "", "property": "", "tenant": "", "timeframe": ""}


def test_paraphrase_collapses_to_same_key():
    # Filler/phrasing differences must not change the key.
    k1 = cache._key("What is the total revenue in 2025?", _REV_2025)
    k2 = cache._key("total revenue across all properties and tenants for the year 2025", _REV_2025)
    assert k1 == k2


def test_different_subject_stays_distinct():
    b15 = {"metric": "revenue", "property": "Building 15", "tenant": "", "timeframe": "2025"}
    b16 = {"metric": "revenue", "property": "Building 16", "tenant": "", "timeframe": "2025"}
    assert cache._key("revenue of the building in 2025", b15) != cache._key("revenue of the building in 2025", b16)


def test_grouping_changes_the_key():
    # A "by month" breakdown must not collide with the plain total.
    assert cache._key("revenue in 2025", _REV_2025) != cache._key("revenue by month in 2025", _REV_2025)


def test_superlative_dimensions_do_not_collide():
    assert cache._key("which building is most profitable?", _EMPTY) != \
           cache._key("which quarter is most profitable?", _EMPTY)


def test_metric_synonyms_collapse():
    a = {"metric": "P&L", "property": "", "tenant": "", "timeframe": "2024"}
    b = {"metric": "profit", "property": "", "tenant": "", "timeframe": "2024"}
    assert cache._key("total P&L in 2024", a) == cache._key("total profit in 2024", b)


def test_get_set_roundtrip():
    cache.clear()
    assert cache.get("revenue in 2025", _REV_2025) is None
    cache.set("revenue in 2025", {"answer": "x"}, _REV_2025)
    assert cache.get("total revenue for 2025", _REV_2025) == {"answer": "x"}  # paraphrase hits
    cache.clear()
