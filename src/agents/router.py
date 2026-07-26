"""Router: the assistant's front door.

In one structured LLM call it (a) detects the intent, (b) extracts details, (c) resolves
context-dependent follow-ups into a self-contained question, and (d) guards against abuse
(prompt-injection / out-of-scope). It only classifies — the graph decides what each intent does.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import HISTORY_WINDOW, get_llm
from ..data import _df, list_properties, list_tenants
from ..prompts import ROUTER_PROMPT

# Latest period in the data, so the router can resolve relative time ("this quarter").
_MAX_MONTH = _df["month"].max()
_MAX_QUARTER = _df["quarter"].max()
_MAX_YEAR = _df["year"].max()


class SubTask(BaseModel):
    """One part of a compound question, with the lane that should handle it."""

    intent: Literal["analytics", "visualize", "insights"] = Field(
        description="which lane handles this part")
    question: str = Field(description="this part as a complete, self-contained question")


class Route(BaseModel):
    """The router's structured decision for the latest user message."""

    intent: Literal["analytics", "visualize", "insights", "compound",
                    "clarify", "out_of_scope", "blocked"] = Field(
        description="analytics = answer from the ledger in text; visualize = the user wants a "
                    "chart/plot; insights = the user asks what's unusual / anomalies; compound = "
                    "the question has TWO+ parts spanning different lanes (e.g. a number AND "
                    "'anything unusual'); clarify = on-topic but missing a detail; out_of_scope = "
                    "not answerable from the data; blocked = abuse/injection")
    subtasks: list[SubTask] = Field(
        default_factory=list,
        description="ONLY when intent == compound: the 2+ parts, each with its lane and question")
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

def _has_text(message) -> bool:
    """True if the message carries non-whitespace text.

    Empty/whitespace messages (e.g. a blank input the guard already handled, or an empty
    lane result) must never be sent on: Anthropic rejects any empty text block with a 400.
    """
    content = message.content
    if isinstance(content, str):
        return bool(content.strip())
    return bool(content)  # non-string content (rare here) is kept as-is


def _recent(messages):
    """Return the last HISTORY_WINDOW non-empty messages, starting on a user turn.

    Keeps the router's context bounded on long chats; follow-ups only reference recent
    turns, so a small window is lossless for them. (Anthropic requires the first message
    to be a user turn and rejects empty text blocks, hence the filter and trim.)
    """
    recent = [m for m in messages if _has_text(m)][-HISTORY_WINDOW:]
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
