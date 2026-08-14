from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from mcp_surface import DAILY_TOOL_NAMES, INTERNAL_CALLABLE_TOOL_NAMES, OmbreFastMCP


async def main() -> None:
    assert "list_handoff_scenes" in INTERNAL_CALLABLE_TOOL_NAMES
    assert "list_handoff_scenes" not in DAILY_TOOL_NAMES
    mcp = OmbreFastMCP("surface-check")

    listed = [SimpleNamespace(name="recall_memory"), SimpleNamespace(name="list_handoff_scenes")]
    with patch.object(FastMCP, "list_tools", AsyncMock(return_value=listed)):
        visible = await mcp.list_tools()
    assert [tool.name for tool in visible] == ["recall_memory"]

    with patch.object(FastMCP, "call_tool", AsyncMock(return_value="ok")) as parent_call:
        assert await mcp.call_tool("list_handoff_scenes", {"limit": 3}) == "ok"
        parent_call.assert_awaited_once_with("list_handoff_scenes", {"limit": 3})

    try:
        await mcp.call_tool("retired_tool", {})
    except ToolError:
        pass
    else:
        raise AssertionError("unknown tools must remain blocked")

    status = mcp.surface_status()
    assert "list_handoff_scenes" not in status["advertised_tool_names"]
    assert "list_handoff_scenes" in status["internal_callable_tool_names"]
    print("internal MCP surface checks passed")


if __name__ == "__main__":
    asyncio.run(main())
