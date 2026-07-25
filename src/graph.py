"""The top-level assistant graph.

    router ─┬─ analytics    → the SQL analyst answers from the ledger
            ├─ clarify      → ask a follow-up (remembered on the next turn)
            ├─ out_of_scope → a fixed, polite decline
            └─ blocked      → a fixed, firm refusal (abuse / prompt-injection)

A checkpointer + per-session thread_id give the assistant memory, so follow-ups and
multi-turn clarifications resolve against earlier turns. Out-of-scope and blocked replies
are static text on purpose — no LLM is invoked for them, so they can't be manipulated.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents.router import classify
from .agents.sql_analyst import ask_sql
from .state import AppState

OUT_OF_SCOPE_MSG = (
    "I can only help with questions about your property portfolio's financial data — "
    "P&L, revenue, expenses, tenants, properties, and time-period comparisons. "
    "I can't help with that one."
)
BLOCKED_MSG = "I can't help with that request."


def _latest_user(messages) -> str:
    """Return the content of the most recent user message."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def _route(state: AppState) -> dict:
    """Classify the latest message and record the routing decision."""
    route = classify(state["messages"])
    return {
        "intent": route.intent,
        "standalone_question": route.standalone_question,
        "entities": {"property": route.property, "tenant": route.tenant,
                     "timeframe": route.timeframe, "metric": route.metric},
        "clarification": route.clarification,
        "route_reason": route.reason,
    }


def _analytics(state: AppState) -> dict:
    """Answer from the ledger via the SQL analyst."""
    question = state["standalone_question"] or _latest_user(state["messages"])
    result = ask_sql(question)
    return {"answer": result["answer"], "reasoning": result["reasoning"],
            "sql": result["sql"], "grounded": result["grounded"],
            "messages": [AIMessage(result["answer"])]}


def _clarify(state: AppState) -> dict:
    """Ask the router's follow-up question."""
    message = state.get("clarification") or "Could you give me a bit more detail?"
    return {"answer": message, "messages": [AIMessage(message)]}


def _out_of_scope(state: AppState) -> dict:
    """Politely decline a benign question the data can't answer."""
    return {"answer": OUT_OF_SCOPE_MSG, "messages": [AIMessage(OUT_OF_SCOPE_MSG)]}


def _blocked(state: AppState) -> dict:
    """Firmly refuse an abusive or manipulative request."""
    return {"answer": BLOCKED_MSG, "messages": [AIMessage(BLOCKED_MSG)]}


_builder = StateGraph(AppState)
_builder.add_node("route", _route)
_builder.add_node("analytics", _analytics)
_builder.add_node("clarify", _clarify)
_builder.add_node("out_of_scope", _out_of_scope)
_builder.add_node("blocked", _blocked)
_builder.add_edge(START, "route")
_builder.add_conditional_edges(
    "route", lambda state: state["intent"],
    {"analytics": "analytics", "clarify": "clarify",
     "out_of_scope": "out_of_scope", "blocked": "blocked"},
)
for _lane in ("analytics", "clarify", "out_of_scope", "blocked"):
    _builder.add_edge(_lane, END)

# MemorySaver keeps per-thread state in memory for the life of the process.
assistant = _builder.compile(checkpointer=MemorySaver())


def ask(question: str, thread_id: str = "default") -> dict:
    """Ask the assistant a question within a conversation.

    Messages under the same `thread_id` share memory, so follow-ups and multi-turn
    clarifications resolve against earlier turns. Returns the answer plus routing details.
    """
    config = {"configurable": {"thread_id": thread_id}}
    final = assistant.invoke({"messages": [HumanMessage(question)]}, config)
    return {
        "answer": final["answer"],
        "intent": final["intent"],
        "standalone_question": final.get("standalone_question", ""),
        "entities": final.get("entities", {}),
        "reason": final.get("route_reason", ""),
        "sql": final.get("sql", ""),
        "grounded": final.get("grounded"),
    }
