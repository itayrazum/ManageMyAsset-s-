"""Shared state for the top-level assistant graph.

`messages` accumulates the whole conversation (via the add_messages reducer) so the
router can resolve follow-ups against earlier turns; combined with a checkpointer and a
per-session thread_id, this gives the assistant memory across turns.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AppState(TypedDict):
    """State passed between the router and the lane nodes."""

    messages: Annotated[list, add_messages]  # full conversation (memory + multi-turn clarify)
    intent: str                # analytics | clarify | out_of_scope | blocked
    standalone_question: str   # latest request rewritten as a self-contained question
    entities: dict             # extracted details (property / tenant / timeframe / metric)
    clarification: str         # follow-up question to ask, when intent == clarify
    route_reason: str          # why the router chose this lane
    answer: str                # final response for this turn
    reasoning: str             # analytics only: the SQL agent's reasoning
    sql: str                   # analytics only: the SQL that was run
    grounded: bool             # analytics only: grounding-check result
    cached: bool               # analytics only: whether the answer came from the cache
