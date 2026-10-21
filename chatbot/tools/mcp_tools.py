import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from chatbot.config import Settings

logger = logging.getLogger(__name__)


def build_server_config(settings: Settings) -> dict[str, dict[str, Any]]:
    """Build the MCP server map. A server is only included if its settings are present."""
    servers: dict[str, dict[str, Any]] = {}

    if settings.calculator_server_path:
        servers["Calculator"] = {
            "transport": "stdio",
            "command": settings.uv_command,
            "args": ["run", "fastmcp", "run", str(settings.calculator_server_path)],
        }

    if settings.expense_tracker_url:
        expense_config: dict[str, Any] = {
            "transport": "http",
            "url": settings.expense_tracker_url,
        }
        if settings.horizon_access_token:
            token = settings.horizon_access_token.get_secret_value()
            expense_config["headers"] = {"Authorization": f"Bearer {token}"}
        servers["Expense Tracker"] = expense_config

    if settings.manim_python_path and settings.manim_server_path:
        manim_config: dict[str, Any] = {
            "transport": "stdio",
            "command": str(settings.manim_python_path),
            "args": [str(settings.manim_server_path)],
        }
        if settings.manim_executable:
            manim_config["env"] = {"MANIM_EXECUTABLE": str(settings.manim_executable)}
        servers["manim-server"] = manim_config

    return servers


async def load_mcp_tools(settings: Settings) -> list[BaseTool]:
    """Load tools server by server so one broken server does not stop the chatbot."""
    servers = build_server_config(settings)
    if not servers:
        logger.warning("No MCP servers configured, running with local tools only")
        return []

    client = MultiServerMCPClient(servers)
    tools: list[BaseTool] = []

    for name in servers:
        try:
            server_tools = await client.get_tools(server_name=name)
        except Exception:
            logger.exception("Skipping MCP server '%s': could not load tools", name)
            continue

        logger.info(
            "MCP server '%s' loaded: %s", name, [tool.name for tool in server_tools]
        )
        tools.extend(server_tools)

    return tools
