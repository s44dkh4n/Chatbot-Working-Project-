import argparse
import asyncio
import logging

from groq import APIStatusError
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from chatbot.config import get_settings
from chatbot.graph import build_graph
from chatbot.llm import create_llm
from chatbot.tools import load_all_tools

EXIT_COMMANDS = {"exit", "quit"}

logger = logging.getLogger(__name__)


async def main(thread_id: str) -> None:
    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    tools = await load_all_tools(settings)
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}")

    graph = build_graph(create_llm(settings), tools, settings.max_history_tokens)

    async with AsyncSqliteSaver.from_conn_string(str(settings.db_path)) as checkpointer:
        chatbot = graph.compile(checkpointer=checkpointer)
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {"thread_id": thread_id},
            "run_name": "chat_session",
        }

        while True:
            prompt = (await asyncio.to_thread(input, "\nYou: ")).strip()
            if prompt.lower() in EXIT_COMMANDS:
                break
            if not prompt:
                continue

            try:
                result = await chatbot.ainvoke(
                    {"messages": [HumanMessage(prompt)]}, config=config
                )
            except APIStatusError as error:
                # Keep the chat loop alive when Groq rejects a request (413, 429, 5xx)
                logger.error(
                    "LLM request failed with status %s: %s",
                    error.status_code,
                    error.message,
                )
                print(
                    f"\nRequest failed with status {error.status_code}. "
                    "Try a shorter request or a new thread id."
                )
                continue

            print("\nAssistant:", result["messages"][-1].content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async LangGraph chatbot with MCP tools")
    parser.add_argument("--thread-id", default="thread_1", help="Conversation thread to resume")
    parser.add_argument("--verbose", action="store_true", help="Show INFO logs")
    args = parser.parse_args()

    # WARNING hides routine logs from httpx, groq, mcp and the graph, but keeps failures
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(main(args.thread_id))



"add these expenses: 1. On August 14, 2026, I spent $4.75 on food for a morning cold brew coffee. 2. On August 22, 2026, I paid $65.20 in transport expenses for a gas station refill. 3. On August 29, 2026, I spent $112.80 on groceries for my weekly supermarket run. 4. On September 20, 2026, I spent $32.00 on entertainment for an IMAX cinema ticket and popcorn. 5. On September 28, 2026, I purchased running sneakers under shopping for $79.99. create Notes for these by yourself."