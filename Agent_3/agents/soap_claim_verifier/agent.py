"""Expose root_agent for `uv run adk web agents` (subdir layout required by ADK)."""

from __future__ import annotations

from ...claim_agent import root_agent

__all__ = ["root_agent"]
