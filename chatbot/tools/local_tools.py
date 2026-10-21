import aiohttp
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.tools import tool

from chatbot.config import get_settings

ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"

@tool
async def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol using Alpha Vantage."""
    api_key = get_settings().alphavantage_api_key
    if api_key is None:
        return {"error": "ALPHAVANTAGE_API_KEY is not configured"}

    # Params are passed separately so the key is URL-encoded and kept out of logged URLs
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key.get_secret_value(),
    }
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(ALPHAVANTAGE_URL, params=params) as response:
            response.raise_for_status()
            return await response.json()


search_tool = DuckDuckGoSearchRun(region="us-en")

LOCAL_TOOLS = [get_stock_price, search_tool]
