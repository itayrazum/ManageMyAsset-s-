"""Pure data-access functions over the property ledger (no LLM / LangChain here).

Each row of the ledger is a revenue (positive `profit`) or expense (negative
`profit`) line, tagged with property, tenant, ledger category, and a time period.
Net P&L for any slice is simply the sum of its `profit` column. These functions
are deterministic and unit-testable on their own; `tools.py` wraps them for the LLM.
"""

from typing import Optional

import pandas as pd

from .config import DATA_PATH

# The ledger is small and read-only, so load it once at import time.
_df = pd.read_parquet(DATA_PATH)


def _resolve(value: str, options) -> Optional[str]:
    """Match a user-supplied name to a known value, case-insensitively.

    Returns the canonical value (e.g. 'Building 17') or None if there is no match.
    """
    if value is None:
        return None
    for option in options:
        if option.lower() == value.strip().lower():
            return option
    return None


def _filter(property_name=None, tenant_name=None, year=None,
            quarter=None, month=None, ledger_type=None) -> pd.DataFrame:
    """Return the ledger rows matching every provided (non-None) filter."""
    df = _df
    if property_name is not None:
        df = df[df["property_name"] == property_name]
    if tenant_name is not None:
        df = df[df["tenant_name"] == tenant_name]
    if year is not None:
        df = df[df["year"] == str(year)]
    if quarter is not None:
        df = df[df["quarter"] == quarter]
    if month is not None:
        df = df[df["month"] == month]
    if ledger_type is not None:
        df = df[df["ledger_type"] == ledger_type]
    return df


def list_properties() -> list[str]:
    """List the names of all properties in the portfolio."""
    return sorted(_df["property_name"].dropna().unique().tolist())


def list_tenants() -> list[str]:
    """List the names of all tenants in the portfolio."""
    return sorted(_df["tenant_name"].dropna().unique().tolist())


def calculate_pnl(property_name: Optional[str] = None,
                  tenant_name: Optional[str] = None,
                  year: Optional[str] = None,
                  quarter: Optional[str] = None,
                  month: Optional[str] = None,
                  ledger_type: Optional[str] = None) -> dict:
    """Calculate net profit & loss (the sum of `profit`) for the given filters.

    All filters are optional; leave one out to include every value for that field.
    Formats: year='2024', quarter='2024-Q1', month='2024-M01',
    ledger_type='revenue' or 'expenses'. Returns the net P&L and the row count,
    or an error message if a named property/tenant is not found.
    """
    if property_name is not None:
        match = _resolve(property_name, _df["property_name"].dropna().unique())
        if match is None:
            return {"error": f"Property '{property_name}' not found."}
        property_name = match

    if tenant_name is not None:
        match = _resolve(tenant_name, _df["tenant_name"].dropna().unique())
        if match is None:
            return {"error": f"Tenant '{tenant_name}' not found."}
        tenant_name = match

    rows = _filter(property_name, tenant_name, year, quarter, month, ledger_type)
    if rows.empty:
        return {"error": "No records match those filters."}
    return {"net_pnl": round(float(rows["profit"].sum()), 2), "records": len(rows)}


def top_tenants(limit: int = 5,
                year: Optional[str] = None,
                quarter: Optional[str] = None) -> list[dict]:
    """Rank tenants by net profit, highest first (optionally within a year/quarter)."""
    rows = _filter(year=year, quarter=quarter).dropna(subset=["tenant_name"])
    ranked = rows.groupby("tenant_name")["profit"].sum().sort_values(ascending=False)
    return [{"tenant": t, "net_pnl": round(float(p), 2)} for t, p in ranked.head(limit).items()]


def breakdown_by(dimension: str,
                 property_name: Optional[str] = None,
                 year: Optional[str] = None,
                 quarter: Optional[str] = None) -> list[dict]:
    """Break down net profit by a column, sorted ascending (most negative first).

    `dimension` is one of: property_name, tenant_name, ledger_type, ledger_group,
    ledger_category, month, quarter, year. Useful for questions like
    'what are my biggest expenses?' or 'profit by property?'.
    """
    valid = {"property_name", "tenant_name", "ledger_type", "ledger_group",
             "ledger_category", "month", "quarter", "year"}
    if dimension not in valid:
        return [{"error": f"Invalid dimension. Choose from: {sorted(valid)}"}]

    rows = _filter(property_name=property_name, year=year, quarter=quarter)
    grouped = rows.groupby(dimension)["profit"].sum().sort_values()
    return [{dimension: k, "net_pnl": round(float(v), 2)} for k, v in grouped.items()]
