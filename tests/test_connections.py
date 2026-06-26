"""
test_connections.py — verifies LangSmith tracing and Groq connectivity.

Run with: python -m tests.test_connections
Or with pytest: pytest tests/test_connections.py -v
"""

import os
from src.config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage


def test_env_vars_are_set():
    """
    Confirms all required environment variables are present.
    Pure config check — no API calls made.
    """
    required_vars = [
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
        "GROQ_API_KEY",
    ]

    print("Checking environment variables...\n")
    all_good = True

    for var in required_vars:
        value = os.getenv(var)
        if value:
            display = value[:8] + "..." if "KEY" in var else value
            print(f"  ✓ {var} = {display}")
        else:
            print(f"  ✗ {var} is NOT set — check your .env file")
            all_good = False

    return all_good


def test_llm_call_is_traced():
    """
    Makes a real LLM call and confirms a trace appears in LangSmith.

    What's happening under the hood:
    1. get_llm() returns a ChatGroq instance (a LangChain object)
    2. .invoke() sends the messages to Groq's API
    3. Because LANGCHAIN_TRACING_V2=true, LangChain automatically
       sends the trace (inputs, outputs, latency, token count) to LangSmith
    4. You don't write any tracing code — it's automatic instrumentation

    Interview note: "instrumentation" means adding observability to code
    without changing its core logic.
    """
    llm = get_llm()

    messages = [
        SystemMessage(content="You are a concise financial analyst."),
        HumanMessage(content="In one sentence, what does NVDA do as a company?"),
    ]

    print("Calling LLM...")
    response = llm.invoke(messages)

    print(f"\nResponse: {response.content}")
    print(f"\nToken usage: {response.usage_metadata}")
    print("\n✓ Check smith.langchain.com → finsight to see the trace!")

    return response


if __name__ == "__main__":
    print("=" * 50)
    print("FinSight — connection check")
    print("=" * 50 + "\n")

    config_ok = test_env_vars_are_set()

    if not config_ok:
        print("\n⚠ Fix the missing env vars above before continuing.")
        print("Copy .env.example to .env and fill in your keys.")
        exit(1)

    print("\nMaking a test LLM call...\n")
    test_llm_call_is_traced()

    print("\n" + "=" * 50)
    print("All checks passed.")
    print("Verify the trace at smith.langchain.com")
    print("=" * 50)