"""
config.py — central place for all configuration.

Every other file imports from here instead of reading env vars directly.
This way if you rename a variable, you change it in one place.
"""

import os
from dotenv import load_dotenv

# load_dotenv() reads your .env file and sets the variables as environment vars.
# It does nothing if the variables are already set (safe to call multiple times).
load_dotenv()


# ── LangSmith ─────────────────────────────────────────────────────────────────
# These three env vars are all LangSmith needs to start tracing.
# You don't import LangSmith anywhere — setting these vars is enough.
# LangChain checks for them automatically on every LLM call.
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "alphalens")


# ── LLM provider ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENV = os.getenv("ENV", "development")


# ── Data sources ──────────────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "alphalens")


def get_llm():
    """
    Returns the right LLM based on your environment.

    - development: uses Groq (free, fast, open-source Llama 3)
    - production:  also uses Groq by default (can swap to OpenAI if needed)

    Why a function instead of a global variable?
    Because you might want to call this multiple times with different
    settings later (e.g. a faster model for simple tasks, slower for synthesis).

    Interview note: this pattern is called a factory function.
    """
    from langchain_groq import ChatGroq

    # llama3-70b-8192 = Llama 3 70B with 8192 token context window
    # 70B parameters makes it smart enough for financial reasoning
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,       # 0 = deterministic output (important for structured data)
        api_key=GROQ_API_KEY,
    )