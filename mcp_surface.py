from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError


DAILY_TOOL_NAMES: tuple[str, ...] = (
    "recall_memory",
    "read_memory",
    "write_scene",
    "edit_scene",
    "set_scene_status",
    "annotate",
    "close_window",
    "revise_window_shadow",
    "narrative_revision_inbox",
    "review_narrative_revision",
    "publish_narrative",
    "read_diary",
    "write_diary",
    "revise_diary",
    "delete_diary",
    "comment_diary",
)

class OmbreFastMCP(FastMCP):
    """FastMCP with one exact daily surface.

    Legacy bucket formats and internal HTTP/Python compatibility remain
    available, but retired MCP names are neither advertised nor callable.
    """

    def __init__(self, *args: Any, config: dict[str, Any] | None = None, **kwargs: Any):
        _ = config
        self.tool_surface = "daily"
        super().__init__(*args, **kwargs)

    async def list_tools(self):
        tools = await super().list_tools()
        by_name = {tool.name: tool for tool in tools}
        return [by_name[name] for name in DAILY_TOOL_NAMES if name in by_name]

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        if name not in DAILY_TOOL_NAMES:
            allowed = ", ".join(DAILY_TOOL_NAMES)
            raise ToolError(
                f"Tool {name!r} is retired on the daily surface. "
                f"Use one of: {allowed}."
            )
        return await super().call_tool(name, arguments)

    def surface_status(self) -> dict[str, Any]:
        return {
            "mode": "daily",
            "advertised_scope": "exact_daily_surface",
            "callable_scope": "exact_daily_surface",
            "advertised_tool_names": list(DAILY_TOOL_NAMES),
            "callable_tool_names": list(DAILY_TOOL_NAMES),
            "daily_tool_names": list(DAILY_TOOL_NAMES),
            "compatibility_aliases_callable": False,
        }
