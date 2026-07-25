"""Shared configuration: environment variables, paths, and the LLM factory."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root is one level above this file's folder (src/).
ROOT = Path(__file__).resolve().parent.parent

# Load API keys and settings from the project-root .env file.
load_dotenv(ROOT / ".env")

# The cleaned property ledger produced by the EDA notebook.
DATA_PATH = ROOT / "data" / "property_ledger.parquet"


def _clean(value: str) -> str:
    """Strip whitespace and stray surrounding quotes from an env value."""
    return (value or "").strip().strip("'\"")


# API keys. The Anthropic key may be provided as CLAUDE_API or the standard ANTHROPIC_API_KEY.
OPENAI_API_KEY = _clean(os.getenv("OPENAI_API_KEY"))
ANTHROPIC_API_KEY = _clean(os.getenv("CLAUDE_API") or os.getenv("ANTHROPIC_API_KEY"))

# Which provider the agents use: 'anthropic' or 'openai'. Swap via .env.
LLM_PROVIDER = _clean(os.getenv("LLM_PROVIDER")) or "anthropic"
OPENAI_MODEL = _clean(os.getenv("OPENAI_MODEL")) or "gpt-4o-mini"
ANTHROPIC_MODEL = _clean(os.getenv("ANTHROPIC_MODEL")) or "claude-haiku-4-5"

# The responder writes prose, so a little warmth reads more naturally than temp 0.
RESPONDER_TEMPERATURE = float(_clean(os.getenv("RESPONDER_TEMPERATURE")) or "0.3")

# Optional evaluator-optimizer: an LLM judge reviews the query before answering.
USE_JUDGE = _clean(os.getenv("USE_JUDGE")).lower() in ("1", "true", "yes")

# Router memory: how many recent messages it sees (~2 messages per turn, so 8 ≈ 4 turns).
HISTORY_WINDOW = int(_clean(os.getenv("HISTORY_WINDOW")) or "8")


def get_llm(temperature: float = 0):
    """Return the chat model for the configured provider.

    Keeping this in one place lets every agent stay provider-agnostic, so we can
    switch between Anthropic and OpenAI from .env without touching agent code.
    Note: Claude Sonnet 5 rejects `temperature`; Haiku 4.5 (our default) accepts it.
    """
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=ANTHROPIC_MODEL, temperature=temperature,
                             api_key=ANTHROPIC_API_KEY)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature,
                      api_key=OPENAI_API_KEY)
