"""Tests for the ledger entity lists (src/data.py)."""

from src.data import list_properties, list_tenants


def test_lists_the_five_properties():
    props = list_properties()
    assert len(props) == 5
    assert "Building 17" in props


def test_lists_the_eighteen_tenants():
    tenants = list_tenants()
    assert len(tenants) == 18
    assert all(t.startswith("Tenant ") for t in tenants)
