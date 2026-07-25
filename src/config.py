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


# API keys. The Anthropic key is stored as CLAUDE_API in .env.
OPENAI_API_KEY = _clean(os.getenv("OPENAI_API_KEY"))
ANTHROPIC_API_KEY = _clean(os.getenv("CLAUDE_API"))

# Which provider the agents use: 'anthropic' or 'openai'. Swap via .env.
LLM_PROVIDER = _clean(os.getenv("LLM_PROVIDER")) or "anthropic"
OPENAI_MODEL = _clean(os.getenv("OPENAI_MODEL")) or "gpt-4o-mini"
ANTHROPIC_MODEL = _clean(os.getenv("ANTHROPIC_MODEL")) or "claude-haiku-4-5"


def get_llm(temperature: float = 0):
    """Return the chat model for the configured provider.

    Keeping this in one place lets every agent stay provider-agnostic, so we can
    switch between Anthropic and OpenAI from .env without touching agent code.
    """
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Claude 5 models no longer accept a temperature parameter.
        return ChatAnthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature,
                      api_key=OPENAI_API_KEY)
