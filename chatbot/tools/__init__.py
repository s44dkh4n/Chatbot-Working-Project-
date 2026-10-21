from langchain_core.tools import BaseTool

from chatbot.config import Settings
from chatbot.tools.local_tools import LOCAL_TOOLS
from chatbot.tools.mcp_tools import load_mcp_tools
from chatbot.tools.rag_tools import build_rag_tools



async def load_all_tools(settings: Settings) -> list[BaseTool]:
    tools = [*LOCAL_TOOLS, *build_rag_tools(settings), *await load_mcp_tools(settings)]

    # Two tools with the same name make the LLM and ToolNode ambiguous, so fail early
    names = [tool.name for tool in tools]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"Duplicate tool names across local and MCP tools: {duplicates}")

    return tools
