"""Anomaly detection over the ledger (no LLM).

Finds unusual monthly movements per ledger category. Each category's monthly totals are
standardized *within that category*, so a small category can be flagged as readily as a big
one; an Isolation Forest then scores the (level, month-over-month) features, and sign flips
(a normally-positive line going negative, or vice-versa) are added deterministically. The LLM
only narrates what this model finds - it does none of the detection itself.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import DATA_PATH

_df = pd.read_parquet(DATA_PATH)
_MIN_MONTHS = 4  # need a few points before "unusual" means anything


def extract_period(text: str) -> str:
    """Pull a time scope ('2024', '2025-Q1') out of free text, or '' if none is named."""
    if not text:
        return ""
    year = re.search(r"\b(20\d{2})\b", text)
    quarter = re.search(r"[Qq]\s*([1-4])", text)
    if year and quarter:
        return f"{year.group(1)}-Q{quarter.group(1)}"
    if year:
        return year.group(1)
    return ""


def _in_period(months: pd.Series, period: str) -> pd.Series:
    """Boolean mask for the 'YYYY-MNN' month strings that fall inside `period`."""
    year, _, quarter = period.partition("-Q")
    mask = months.str.startswith(f"{year}-")
    if quarter:
        q = int(quarter)
        month_num = months.str.split("-M").str[1].astype(int)
        mask &= month_num.between(3 * (q - 1) + 1, 3 * q)
    return mask


def _monthly(property: Optional[str], tenant: Optional[str],
             period: Optional[str] = None) -> pd.DataFrame:
    """Monthly total per ledger category, within the requested scope."""
    d = _df
    if property:
        d = d[d["property_name"] == property]
    if tenant:
        d = d[d["tenant_name"] == tenant]
    if period:
        d = d[_in_period(d["month"], period)]
    return (d.groupby(["ledger_category", "month"], as_index=False)["profit"].sum()
             .sort_values(["ledger_category", "month"]))


def detect_anomalies(top_n: int = 6, property: Optional[str] = None,
                     tenant: Optional[str] = None, period: Optional[str] = None) -> list[dict]:
    """Return the top unusual monthly movements, each with the numbers to explain it.

    `period` ('2024', '2025-Q1', ...) scopes detection to that window, so the baseline and
    the flagged months both stay inside the period the user asked about.
    """
    monthly = _monthly(property, tenant, period)

    rows = []
    for category, grp in monthly.groupby("ledger_category"):
        values = grp["profit"].to_numpy(dtype=float)
        if len(values) < _MIN_MONTHS or values.std() == 0:
            continue
        mean = values.mean()
        z = (values - mean) / values.std()
        mom = np.diff(values, prepend=values[0])
        z_mom = mom / (mom.std() or 1.0)
        for i, month in enumerate(grp["month"].to_numpy()):
            rows.append({
                "category": category, "month": str(month),
                "value": round(float(values[i]), 2), "typical": round(float(mean), 2),
                "z": float(z[i]), "z_mom": float(z_mom[i]),
                "sign_flip": bool(np.sign(values[i]) != np.sign(mean)),
            })
    if not rows:
        return []

    features = np.array([[r["z"], r["z_mom"]] for r in rows])
    model = IsolationForest(contamination=0.06, random_state=42)
    labels = model.fit_predict(features)      # -1 = anomaly
    scores = model.score_samples(features)    # lower = more anomalous
    for r, label, score in zip(rows, labels, scores):
        # A sign flip only matters if the flipped amount is material, not a rounding blip.
        r["material_flip"] = r["sign_flip"] and abs(r["value"]) > max(100.0, 0.15 * abs(r["typical"]))
        r["severity"] = max(abs(r["z"]), abs(r["z_mom"]))
        # A deviation must be both statistically off (>=2 sigma) AND practically large, so a
        # 5% wobble in a very steady series isn't reported as "unusual".
        material_dev = abs(r["value"] - r["typical"]) > max(500.0, 0.25 * abs(r["typical"]))
        r["report"] = r["material_flip"] or (label == -1 and r["severity"] >= 2.0 and material_dev)

    reported = [r for r in rows if r["report"]]
    reported.sort(key=lambda r: (not r["material_flip"], -r["severity"]))  # flips first, then most severe

    result = []
    for r in reported[:top_n]:
        if r["material_flip"]:
            note = f"sign flip - {r['value']} vs its usual {r['typical']}"
        else:
            note = f"{'well above' if r['value'] > r['typical'] else 'well below'} its usual {r['typical']}"
        result.append({"category": r["category"], "month": r["month"],
                       "value": r["value"], "typical": r["typical"], "note": note})
    return result


def monthly_series(category: str, property: Optional[str] = None,
                   tenant: Optional[str] = None) -> list[dict]:
    """Monthly totals for one ledger category - to see if a flag is a one-off or a trend."""
    d = _df[_df["ledger_category"] == category]
    if property:
        d = d[d["property_name"] == property]
    if tenant:
        d = d[d["tenant_name"] == tenant]
    g = d.groupby("month", as_index=False)["profit"].sum().sort_values("month")
    return [{"month": str(m), "value": round(float(v), 2)}
            for m, v in zip(g["month"], g["profit"])]


def contributors(category: str, month: str, property: Optional[str] = None,
                 tenant: Optional[str] = None) -> list[dict]:
    """Break a category in a given month down by tenant (or property) - who drove it."""
    d = _df[(_df["ledger_category"] == category) & (_df["month"] == month)]
    if property:
        d = d[d["property_name"] == property]
    if tenant:
        d = d[d["tenant_name"] == tenant]
    dim = "tenant_name" if d["tenant_name"].notna().any() else "property_name"
    g = d.dropna(subset=[dim]).groupby(dim, as_index=False)["profit"].sum().sort_values("profit")
    return [{"who": w, "value": round(float(v), 2)} for w, v in zip(g[dim], g["profit"])]
