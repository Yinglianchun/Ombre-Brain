from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError


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
    """Resolve the advertised and callable MCP surface."""

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
    """FastMCP with one bounded daily surface and an explicit admin rollback.

    FastMCP keeps tool listing and tool invocation separate, so filtering only
    ``tools/list`` leaves cached legacy schemas callable.  Daily mode applies
    the same allowlist at dispatch time.  Internal Python helpers and
    Dashboard/admin HTTP routes remain available.  ``legacy``/``admin`` is the
    explicit one-setting rollback that advertises and permits every registered
    MCP tool.
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

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        if self.tool_surface not in _FULL_SURFACE_NAMES and name not in DAILY_TOOL_NAMES:
            allowed = ", ".join(DAILY_TOOL_NAMES)
            raise ToolError(
                f"Tool {name!r} is retired on the daily surface. "
                f"Use one of: {allowed}."
            )
        return await super().call_tool(name, arguments)

    def surface_status(self) -> dict[str, Any]:
        is_daily = self.tool_surface == "daily"
        return {
            "mode": self.tool_surface,
            "advertised_scope": "daily_allowlist" if is_daily else "all_registered",
            "callable_scope": "daily_allowlist" if is_daily else "all_registered",
            "advertised_tool_names": list(DAILY_TOOL_NAMES) if is_daily else None,
            "callable_tool_names": list(DAILY_TOOL_NAMES) if is_daily else None,
            "daily_tool_names": list(DAILY_TOOL_NAMES),
            "compatibility_aliases_callable": not is_daily,
        }
