import logging
import re
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from chatbot.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Tools in this project report expected failures as {"status": "error", ...} or {"error": ...}
ERROR_PATTERN = re.compile(r'^\s*\{\s*"error"\s*:|"status"\s*:\s*"error"')


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def select_recent_turns(
    messages: list[BaseMessage], max_tokens: int
) -> list[BaseMessage]:
    """Keep the current turn plus as many older whole turns as fit in max_tokens."""
    # A turn starts at a HumanMessage and includes the AI and tool messages after it
    # Cutting only at turn starts never separates a tool call from its tool result
    starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if not starts:
        return messages

    # The current turn is always kept, even if it alone is over the budget
    first = starts[-1]
    for start in reversed(starts[:-1]):
        if count_tokens_approximately(messages[start:]) > max_tokens:
            break
        first = start

    return messages[first:]


def _content_text(message: ToolMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # MCP tools return a list of content blocks, so join their text parts
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


def _call_args(messages: list[BaseMessage], tool_call_id: str) -> dict | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["id"] == tool_call_id:
                    return call["args"]
    return None


def log_recent_tool_results(messages: list[BaseMessage]) -> None:
    """Failed tool calls are logged as warnings, successful ones only at INFO level."""
    recent: list[ToolMessage] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        recent.append(message)

    # Walk the results oldest first so the log reads in call order
    for message in reversed(recent):
        text = _content_text(message)
        if message.status == "error" or ERROR_PATTERN.search(text):
            logger.warning(
                "Tool failed: %s args=%s result=%s",
                message.name,
                _call_args(messages, message.tool_call_id),
                text[:300],
            )
        else:
            logger.info("Tool result (%s): %s", message.name, text[:300])


def build_graph(
    llm: BaseChatModel, tools: list[BaseTool], max_history_tokens: int = 2500
) -> StateGraph:
    model = llm.bind_tools(tools)

    async def chat_node(state: ChatState) -> dict:
        log_recent_tool_results(state["messages"])

        # The checkpoint keeps the full history, only the model input is trimmed
        history = select_recent_turns(state["messages"], max_history_tokens)
        dropped = len(state["messages"]) - len(history)
        if dropped:
            logger.info("Trimmed %d older messages from the model input", dropped)

        # System prompt is added per call so it is not saved into the checkpoint history
        messages = [SystemMessage(content=SYSTEM_PROMPT), *history]
        response = await model.ainvoke(messages)

        for tool_call in response.tool_calls:
            logger.info("Tool call: %s args=%s", tool_call["name"], tool_call["args"])

        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    # Exceptions inside a tool become an error message the LLM can read and retry
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph