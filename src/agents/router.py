"""Router: the assistant's front door.

In one structured LLM call it (a) detects the intent, (b) extracts details, (c) resolves
context-dependent follow-ups into a self-contained question, and (d) guards against abuse
(prompt-injection / out-of-scope). It only classifies — the graph decides what each intent does.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import get_llm
from ..data import _df, list_properties, list_tenants
from ..prompts import ROUTER_PROMPT

# Latest period in the data, so the router can resolve relative time ("this quarter").
_MAX_MONTH = _df["month"].max()
_MAX_QUARTER = _df["quarter"].max()
_MAX_YEAR = _df["year"].max()


class Route(BaseModel):
    """The router's structured decision for the latest user message."""

    intent: Literal["analytics", "clarify", "out_of_scope", "blocked"] = Field(
        description="analytics = answerable from the ledger; clarify = on-topic but missing a "
                    "detail; out_of_scope = not answerable from the data; blocked = abuse/injection")
    standalone_question: str = Field(
        default="", description="The latest request rewritten as a self-contained question, "
                               "resolving earlier turns; used for analytics")
    property: str = Field(default="", description="Property/building named, if any")
    tenant: str = Field(default="", description="Tenant named, if any")
    timeframe: str = Field(default="", description="Timeframe named (year/quarter/month), if any")
    metric: str = Field(default="", description="What is asked (revenue, expenses, pnl, comparison, ...)")
    clarification: str = Field(default="", description="A short follow-up question, if intent == clarify")
    reason: str = Field(default="", description="One-line reason for the classification")


_router = get_llm().with_structured_output(Route)

# Keep only the last few messages so the router's context stays bounded on long chats.
# Follow-ups reference recent turns, so a small window is enough (and lossless for them).
_HISTORY_WINDOW = 8


def _recent(messages):
    """Return the last few messages, starting on a user turn (Anthropic requires that)."""
    recent = messages[-_HISTORY_WINDOW:]
    while recent and not isinstance(recent[0], HumanMessage):
        recent = recent[1:]
    return recent


def classify(messages) -> Route:
    """Classify the latest user message, using a recent window of the conversation."""
    system = ROUTER_PROMPT.format(
        properties=", ".join(list_properties()),
        tenants=", ".join(list_tenants()),
        max_month=_MAX_MONTH, max_quarter=_MAX_QUARTER, max_year=_MAX_YEAR,
    )
    return _router.invoke([SystemMessage(system), *_recent(messages)])
