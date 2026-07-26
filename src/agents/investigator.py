"""The Investigator: a tool-using agent for open-ended "what's unusual" investigation.

Unlike the analytics path (a controlled generate->execute pipeline), investigation is
exploratory - the next step depends on what the last one found - so this is a genuine
tool-calling agent. Its tools are all deterministic (an anomaly-detection model and two pandas
drill-downs), so the LLM only decides *which* tool to call and how to phrase the result; every
number comes from a tool.
"""

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ..anomaly import contributors, detect_anomalies, monthly_series
from ..config import get_llm
from ..prompts import INVESTIGATOR_PROMPT


@tool
def find_anomalies(property: str = "", tenant: str = "") -> list:
    """Flag unusual monthly movements in the ledger with an anomaly-detection model.

    Optionally scope to a property (e.g. 'Building 17') or tenant (e.g. 'Tenant 7'); leave blank
    for the whole portfolio. Returns flagged points (category, month, value, typical, note).
    An empty list means nothing stood out.
    """
    return detect_anomalies(property=property or None, tenant=tenant or None)


@tool
def category_history(category: str, property: str = "", tenant: str = "") -> list:
    """The monthly totals for one ledger category - use it to see whether a flagged point is a
    one-off spike or part of a trend. Optionally scope to a property or tenant."""
    return monthly_series(category, property or None, tenant or None)


@tool
def who_drove(category: str, month: str, property: str = "", tenant: str = "") -> list:
    """Break a category in a given month down by tenant (or property) - use it to see who or what
    drove a flagged movement."""
    return contributors(category, month, property or None, tenant or None)


_TOOLS = [find_anomalies, category_history, who_drove]
_agent = create_react_agent(get_llm(), _TOOLS, prompt=INVESTIGATOR_PROMPT)


def investigate(question: str) -> dict:
    """Run the investigation loop; return the final answer and which tools it used."""
    try:
        result = _agent.invoke({"messages": [("user", question)]}, {"recursion_limit": 14})
    except Exception:
        return {"answer": "", "tools_used": [], "error": True}
    tools_used = [call["name"]
                  for message in result["messages"]
                  for call in getattr(message, "tool_calls", None) or []]
    return {"answer": result["messages"][-1].content, "tools_used": tools_used}
