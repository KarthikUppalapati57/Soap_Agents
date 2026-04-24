"""Expose root_agent for `uv run adk web agents` (subdir layout required by ADK)."""

from __future__ import annotations

import sys
from pathlib import Path

# Agent-3 repo root (parent of `agents/`)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claim_agent import root_agent

__all__ = ["root_agent"]
