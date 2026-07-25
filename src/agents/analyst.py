"""The analyst agent: an OpenAI tool-calling agent, built with LangGraph, that
answers natural-language questions about the property ledger via the data tools.
"""

from langgraph.prebuilt import create_react_agent

from ..config import get_llm
from ..prompts import ANALYST_SYSTEM_PROMPT
from ..tools import TOOLS

# temperature=0 for consistent, deterministic analysis.
_llm = get_llm()

# create_react_agent builds a LangGraph tool-calling loop (reason -> call tool -> repeat).
_agent = create_react_agent(_llm, TOOLS, prompt=ANALYST_SYSTEM_PROMPT)


def ask(question: str) -> str:
    """Ask the analyst agent a question and return its final natural-language answer."""
    result = _agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content
