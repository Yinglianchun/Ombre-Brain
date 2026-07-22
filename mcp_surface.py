from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP


DAILY_TOOL_NAMES: tuple[str, ...] = (
    "recall",
    "read_memory",
    "write_scene",
    "annotate",
    "close_window",
    "publish_narrative",
    "read_portrait",
    "publish_portrait",
)

_FULL_SURFACE_NAMES = frozenset({"legacy", "full", "admin"})


def tool_surface_mode(config: dict[str, Any] | None = None) -> str:
    """Resolve the advertised MCP surface without changing callable aliases."""

    config = config or {}
    mcp_config = config.get("mcp", {})
    if not isinstance(mcp_config, dict):
        mcp_config = {}
    requested = str(
        os.environ.get("OMBRE_MCP_TOOL_SURFACE")
        or mcp_config.get("tool_surface")
        or "daily"
    ).strip().lower()
    return requested if requested in _FULL_SURFACE_NAMES else "daily"


class OmbreFastMCP(FastMCP):
    """FastMCP with a small advertised daily surface and hidden legacy calls.

    FastMCP keeps tool listing and tool invocation separate.  We filter only
    ``tools/list`` here: existing Bridge jobs and older clients may continue to
    call registered compatibility tools by their exact names, while an ordinary
    chat model sees only the bounded façade.  ``legacy`` is a one-setting
    rollback that advertises every registered tool again.
    """

    def __init__(self, *args: Any, config: dict[str, Any] | None = None, **kwargs: Any):
        self.tool_surface = tool_surface_mode(config)
        super().__init__(*args, **kwargs)

    async def list_tools(self):
        tools = await super().list_tools()
        if self.tool_surface in _FULL_SURFACE_NAMES:
            return tools
        by_name = {tool.name: tool for tool in tools}
        return [by_name[name] for name in DAILY_TOOL_NAMES if name in by_name]

    def surface_status(self) -> dict[str, Any]:
        is_daily = self.tool_surface == "daily"
        return {
            "mode": self.tool_surface,
            "advertised_scope": "daily_allowlist" if is_daily else "all_registered",
            "advertised_tool_names": list(DAILY_TOOL_NAMES) if is_daily else None,
            "daily_tool_names": list(DAILY_TOOL_NAMES),
            "compatibility_aliases_callable": True,
        }
