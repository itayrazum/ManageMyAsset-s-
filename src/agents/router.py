"""Router: the assistant's front door.

In one structured LLM call it (a) detects the intent, (b) extracts details, (c) resolves
context-dependent follow-ups into a self-contained question, and (d) guards against abuse
(prompt-injection / out-of-scope). It only classifies — the graph decides what each intent does.
"""

from typing import Literal

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from ..config import get_llm
from ..data import list_properties, list_tenants
from ..prompts import ROUTER_PROMPT


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


def classify(messages) -> Route:
    """Classify the latest user message, in the context of the whole conversation."""
    system = ROUTER_PROMPT.format(
        properties=", ".join(list_properties()),
        tenants=", ".join(list_tenants()),
    )
    return _router.invoke([SystemMessage(system), *messages])
