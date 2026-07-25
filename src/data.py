"""Loads the property ledger and exposes the entity lists the router needs.

The analytics path queries the ledger via DuckDB (see agents/sql_analyst.py); this module
just provides the property/tenant names used to ground the router's prompt.
"""

import pandas as pd

from .config import DATA_PATH

# The ledger is small and read-only, so load it once at import time.
_df = pd.read_parquet(DATA_PATH)


def list_properties() -> list[str]:
    """List the names of all properties in the portfolio."""
    return sorted(_df["property_name"].dropna().unique().tolist())


def list_tenants() -> list[str]:
    """List the names of all tenants in the portfolio."""
    return sorted(_df["tenant_name"].dropna().unique().tolist())
