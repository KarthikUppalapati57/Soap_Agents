"""Load `.env` and normalize API key env vars for Google ADK / Gemini."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    load_dotenv(root.parent / ".env")
    key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("gemini_api_key")
    )
    if key:
        os.environ.setdefault("GOOGLE_API_KEY", key)


def require_api_key() -> None:
    load_env()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Missing GOOGLE_API_KEY (or GEMINI_API_KEY / gemini_api_key in .env). "
            "See .env.example in Agent_3."
        )
