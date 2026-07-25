"""Structured answer cache for the analytics lane.

Keyed on the router's extracted entities (metric / property / tenant / timeframe) plus an
"operation signature" — the non-default operation words present in the question (average,
biggest, by-month, count, compare, ...). Plain filler and phrasing ("total", "for the year",
"across all properties") are ignored, so paraphrases of the same computation collapse to one
key, while a different subject (Building 15 vs 16) or operation (total vs by-month, building
vs quarter) stays distinct. That lifts the hit rate without risking a wrong-number false hit.

Namespaced by the data file + active model so either change invalidates old entries.
In-memory (per process); swap the dict for a persistent store (e.g. diskcache) later without
changing this interface — exact match stays conservative: a miss just recomputes.
"""

import hashlib
import re

from .config import ANTHROPIC_MODEL, DATA_PATH, LLM_PROVIDER, OPENAI_MODEL

# Version the cache by the data file and the active model — either change busts old keys.
_DATA_HASH = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()[:12]
_MODEL = ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else OPENAI_MODEL
_NAMESPACE = f"{_DATA_HASH}:{_MODEL}"

# Words that signal a NON-default operation or a grouping dimension. Their presence shapes
# the computation; their absence means a plain total/sum. Everything else (filler, entity
# names, phrasing) is ignored, so paraphrases collapse together.
_OPERATIONS = {
    "average", "avg", "mean",
    "biggest", "largest", "highest", "most", "max", "maximum", "top", "best",
    "smallest", "lowest", "least", "min", "minimum", "worst",
    "rank", "ranking", "list", "which", "who",
    "count", "many", "distinct",
    "share", "percentage", "percent", "ratio", "proportion",
    "compare", "comparison", "versus", "vs", "difference", "change", "growth",
    "higher", "lower",
    "by", "per", "each", "breakdown", "split", "monthly", "quarterly",
    "building", "quarter", "month", "category", "group",
    "unusual", "anomaly", "trend",
}

# Map the router's free-text metric to a canonical token so "P&L" / "profit" / "net P&L"
# (or "income" / "rent") don't split the cache key.
_METRIC_ALIASES = {
    "pnl": ("p&l", "p and l", "pnl", "profit", "loss", "earnings", "net"),
    "revenue": ("revenue", "income", "sales", "turnover", "rent"),
    "expenses": ("expense", "cost", "spend", "outgoing"),
}

_store: dict[str, dict] = {}


def _canonical_metric(metric: str) -> str:
    """Collapse metric synonyms (P&L / profit / income / cost ...) to one token."""
    text = (metric or "").lower()
    for canonical, aliases in _METRIC_ALIASES.items():
        if any(alias in text for alias in aliases):
            return canonical
    return text.strip()


# Portfolio-wide / all-time phrasings mean "no specific subject" — treat them as empty so
# "all properties" and "" share a key.
_ALL_TERMS = {"portfolio", "entire portfolio", "whole portfolio", "everything",
              "all time", "all-time", "2024-2025", "2024 and 2025"}


def _canonical_subject(value: str) -> str:
    """Normalise 'all' / 'portfolio' / 'all time' to empty; otherwise lowercase and strip."""
    text = (value or "").strip().lower()
    if text in _ALL_TERMS or text.startswith("all "):
        return ""
    return text


def _signature(question: str) -> str:
    """The sorted set of operation/dimension words present in the question."""
    # Use a set comprehension, not set(): this module defines a `set` function that
    # would otherwise shadow the builtin here.
    words = {w for w in re.findall(r"[a-z]+", (question or "").lower())}
    return " ".join(sorted(words & _OPERATIONS))


def _key(question: str, entities: dict | None = None) -> str:
    """Build a namespaced key from the entities plus the operation signature."""
    entities = entities or {}
    metric = _canonical_metric(entities.get("metric"))
    prop = _canonical_subject(entities.get("property"))
    tenant = _canonical_subject(entities.get("tenant"))
    timeframe = _canonical_subject(entities.get("timeframe"))
    return (f"{_NAMESPACE}|metric={metric}|property={prop}|tenant={tenant}"
            f"|timeframe={timeframe}|{_signature(question)}")


def get(question: str, entities: dict | None = None):
    """Return the cached result for a question, or None on a miss."""
    return _store.get(_key(question, entities))


def set(question: str, result: dict, entities: dict | None = None) -> None:
    """Store a result for a question."""
    _store[_key(question, entities)] = result


def clear() -> None:
    """Empty the cache (useful in tests and notebooks)."""
    _store.clear()
