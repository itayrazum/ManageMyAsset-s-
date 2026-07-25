"""Simple in-memory answer cache for the analytics lane.

Exact match on the normalized standalone question, namespaced by the data file and the
active model so that changing either invalidates old entries. In-memory only (per process);
the dict can later be swapped for a persistent store (e.g. diskcache) without changing this
interface. Exact match is deliberately conservative: a miss just recomputes, whereas a fuzzy
false hit would serve a wrong financial number.
"""

import hashlib
import re

from .config import ANTHROPIC_MODEL, DATA_PATH, LLM_PROVIDER, OPENAI_MODEL

# Version the cache by the data file and the active model — either change busts old keys.
_DATA_HASH = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()[:12]
_MODEL = ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else OPENAI_MODEL
_NAMESPACE = f"{_DATA_HASH}:{_MODEL}"

_store: dict[str, dict] = {}


def _key(question: str) -> str:
    """Build a namespaced key from a normalized question (lowercased, trimmed, no '?')."""
    normalized = re.sub(r"\s+", " ", question.strip().lower()).rstrip("?").strip()
    return f"{_NAMESPACE}|{normalized}"


def get(question: str):
    """Return the cached result for a question, or None on a miss."""
    return _store.get(_key(question))


def set(question: str, result: dict) -> None:
    """Store a result for a question."""
    _store[_key(question)] = result


def clear() -> None:
    """Empty the cache (useful in tests and notebooks)."""
    _store.clear()
