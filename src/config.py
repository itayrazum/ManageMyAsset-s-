"""Shared configuration: loads environment variables and common paths/settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root is one level above this file's folder (src/).
ROOT = Path(__file__).resolve().parent.parent

# Load OPENAI_API_KEY and other variables from the project-root .env file.
load_dotenv(ROOT / ".env")

# The cleaned property ledger produced by the EDA notebook.
DATA_PATH = ROOT / "data" / "property_ledger.parquet"

# OpenAI chat model used by the agent (override via .env if desired).
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
