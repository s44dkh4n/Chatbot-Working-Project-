from langchain_groq import ChatGroq

from chatbot.config import Settings


def create_llm(settings: Settings) -> ChatGroq:
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        max_retries=settings.llm_max_retries,
        max_tokens=settings.llm_max_tokens,
        reasoning_format="hidden",
    )
