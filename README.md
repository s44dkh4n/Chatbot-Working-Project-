# GenAI MCP Chatbot

Async LangGraph chatbot with Groq-hosted LLMs, local tools, and MCP servers.

## Structure

    main.py                    entry point and chat loop
    chatbot/config.py          typed settings from environment variables
    chatbot/llm.py             LLM factory
    chatbot/prompts.py         system prompt
    chatbot/graph.py           LangGraph state, nodes, and wiring
    chatbot/tools/local_tools.py   stock price, web search
    chatbot/tools/mcp_tools.py     MCP server config and tool loading
    chatbot/tools/__init__.py      combines local and MCP tools

## Setup

    python -m venv .venv
    pip install -r requirements.txt
    cp .env.example .env

Fill in `.env`. Any MCP server with blank settings is skipped.

## Run

    python main.py
    python main.py --thread-id my_thread

Type `exit` to quit. Reusing a thread id resumes that conversation.
