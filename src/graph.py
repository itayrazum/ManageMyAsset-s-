"""The top-level assistant graph.

    router ─┬─ analytics    → the SQL analyst answers from the ledger
            ├─ clarify      → ask a follow-up (remembered on the next turn)
            ├─ out_of_scope → a fixed, polite decline
            └─ blocked      → a fixed, firm refusal (abuse / prompt-injection)

A checkpointer + per-session thread_id give the assistant memory, so follow-ups and
multi-turn clarifications resolve against earlier turns. Out-of-scope and blocked replies
are static text on purpose — no LLM is invoked for them, so they can't be manipulated.
"""

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import cache
from .agents.responder import write_answer, write_caption
from .agents.router import classify
from .agents.investigator import investigate
from .agents.sql_analyst import ask_sql
from .anomaly import detect_anomalies
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


MAX_INPUT_CHARS = 1000


def _validate_input(text: str) -> str:
    """Deterministic guard on the raw input; returns a guidance message if it's unusable.

    Cheap checks that don't need the LLM (empty / whitespace / very long) - the last also
    keeps a stray huge paste from blowing up the context on the public URL.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 'Could you type a question? For example: "What was my P&L in 2024?"'
    if len(stripped) > MAX_INPUT_CHARS:
        return (f"That message is quite long ({len(stripped)} characters). Could you shorten it "
                f"to a specific question about your portfolio?")
    return ""


def _route(state: AppState) -> dict:
    """Guard the input, then classify the latest message and record the routing decision."""
    problem = _validate_input(_latest_user(state["messages"]))
    if problem:  # short-circuit to a helpful clarify, without an LLM call
        return {"intent": "clarify", "clarification": problem,
                "standalone_question": "", "entities": {}, "route_reason": "input guard"}
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
    """Answer from the ledger via the SQL analyst, using the answer cache."""
    question = state["standalone_question"] or _latest_user(state["messages"])
    entities = state.get("entities", {})
    result = cache.get(question, entities)
    cached = result is not None
    if not cached:
        result = ask_sql(question)
        cache.set(question, result, entities)
    return {"answer": result["answer"], "reasoning": result["reasoning"],
            "sql": result["sql"], "grounded": result["grounded"], "cached": cached,
            "messages": [AIMessage(result["answer"])]}


_TIME_DIMENSIONS = ("month", "quarter", "year")


def _chart_spec(rows: list) -> dict | None:
    """Pick a chart type and x/y columns from grouped rows, or None if not chartable."""
    if not rows or len(rows) < 2:  # need at least two points to plot
        return None
    columns = list(rows[0].keys())
    numeric = [c for c in columns
               if isinstance(rows[0].get(c), (int, float)) and not isinstance(rows[0].get(c), bool)]
    dimensions = [c for c in columns if c not in numeric]
    if not numeric or not dimensions:
        return None
    x = dimensions[0]
    y = numeric[0]
    chart_type = "line" if any(t in x.lower() for t in _TIME_DIMENSIONS) else "bar"
    return {"type": chart_type, "x": x, "y": y}


def _visualize(state: AppState) -> dict:
    """Answer with a chart: get grouped data via the SQL analyst, plot it, and add a caption.

    If the result isn't chartable, fall back to the SQL analyst's normal text answer.
    """
    question = state["standalone_question"] or _latest_user(state["messages"])
    result = ask_sql(question)
    spec = _chart_spec(result.get("data") or [])

    chart_rows = None
    if spec:
        df = pd.DataFrame(result["data"])
        # Drop rollup/grand-total rows so one big bar doesn't dwarf the real categories.
        labels = df[spec["x"]].astype(str).str.strip().str.lower()
        df = df[~labels.isin({"total", "all", "grand total", "overall", "none", ""})]
        # Collapse any extra breakdown dimensions to one value per x, sorted for a clean axis.
        agg = df.groupby(spec["x"], as_index=False)[spec["y"]].sum().sort_values(spec["x"])
        if len(agg) >= 2:
            chart_rows = agg.to_dict("records")

    if chart_rows is None:  # not chartable — use the plain text answer
        return {"answer": result["answer"], "reasoning": result["reasoning"],
                "sql": result["sql"], "grounded": result["grounded"],
                "messages": [AIMessage(result["answer"])]}

    # A one-sentence caption from the aggregated points (not the raw per-row breakdown).
    caption = write_caption(question, chart_rows)
    return {"answer": caption, "reasoning": result["reasoning"], "sql": result["sql"],
            "grounded": result["grounded"], "chart_data": chart_rows,
            "chart_x": spec["x"], "chart_y": spec["y"], "chart_type": spec["type"],
            "messages": [AIMessage(caption)]}


def _insights(state: AppState) -> dict:
    """Investigate what's unusual via a tool-using agent (anomaly model + the SQL analyst)."""
    question = state["standalone_question"] or _latest_user(state["messages"])
    result = investigate(question)
    if result.get("error") or not result["answer"]:
        # Fallback: report the raw anomalies if the agent loop didn't produce an answer.
        entities = state.get("entities", {})
        anomalies = detect_anomalies(property=entities.get("property") or None,
                                     tenant=entities.get("tenant") or None)
        note = (f"An anomaly-detection model flagged: {anomalies}. Summarize the notable ones "
                "briefly." if anomalies else "Nothing unusual stood out; say so in one sentence.")
        answer = write_answer(question, note=note)
        return {"answer": answer, "reasoning": "Anomaly detection (fallback)",
                "messages": [AIMessage(answer)]}
    tools = result["tools_used"]
    reasoning = "Investigated with tools: " + (", ".join(tools) if tools else "none")
    return {"answer": result["answer"], "reasoning": reasoning,
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
_builder.add_node("visualize", _visualize)
_builder.add_node("insights", _insights)
_builder.add_node("clarify", _clarify)
_builder.add_node("out_of_scope", _out_of_scope)
_builder.add_node("blocked", _blocked)
_builder.add_edge(START, "route")
_builder.add_conditional_edges(
    "route", lambda state: state["intent"],
    {"analytics": "analytics", "visualize": "visualize", "insights": "insights",
     "clarify": "clarify", "out_of_scope": "out_of_scope", "blocked": "blocked"},
)
for _lane in ("analytics", "visualize", "insights", "clarify", "out_of_scope", "blocked"):
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
        "reasoning": final.get("reasoning", ""),
        "sql": final.get("sql", ""),
        "grounded": final.get("grounded"),
        "cached": final.get("cached", False),
        "chart_data": final.get("chart_data"),
        "chart_x": final.get("chart_x"),
        "chart_y": final.get("chart_y"),
        "chart_type": final.get("chart_type"),
    }
