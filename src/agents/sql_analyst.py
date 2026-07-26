"""Text-to-SQL analyst agent (DuckDB), built as an explicit LangGraph pipeline.

Flow:  generate -> execute -> [judge] -> respond -> check
- generate: the LLM writes its reasoning and ONE read-only SQL query (structured output).
- execute:  DuckDB runs the query; on a SQL error the graph loops back to generate so
            the agent can self-correct (up to MAX_ATTEMPTS).
- judge:    OPTIONAL (config.USE_JUDGE) — an LLM reviews the query and can send it back
            to generate with feedback (the evaluator-optimizer pattern).
- respond:  the shared Responder phrases the final answer from the returned rows.
- check:    a deterministic grounding check flags any answer number not backed by the data.

All arithmetic happens in SQL — the model never computes numbers itself. The agent returns
{reasoning, sql, answer, data, grounded, unsupported, judge_ok, judge_feedback}.

Data access: the property-ledger parquet is exposed as a read-only DuckDB view named
`ledger` (in-memory; the parquet file is never modified).
"""

import logging
import re
from typing import TypedDict

import duckdb
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..checks import check_grounding
from ..config import DATA_PATH, USE_JUDGE, get_llm
from ..logging_config import snippet
from ..prompts import SQL_GENERATE_PROMPT, SQL_JUDGE_PROMPT
from .responder import write_answer

logger = logging.getLogger(__name__)

# --- DuckDB setup ------------------------------------------------------------

_con = duckdb.connect(":memory:")
_con.execute(f"CREATE VIEW ledger AS SELECT * FROM read_parquet('{DATA_PATH.as_posix()}')")

# Read the time bounds from the data so the prompt stays correct if the data changes.
_MIN_MONTH, _MAX_MONTH, _MAX_QUARTER = _con.execute(
    "SELECT MIN(month), MAX(month), MAX(quarter) FROM ledger"
).fetchone()

# Keywords that would modify data or touch the filesystem — never allowed.
_FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create", "replace",
              "attach", "detach", "copy", "export", "install", "load", "pragma", "set")


def _is_safe(query: str) -> bool:
    """Return True only if the query is a single read-only SELECT/WITH statement."""
    q = query.strip().rstrip(";")
    low = q.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return False
    if ";" in q:  # block stacked statements like `SELECT ...; DROP ...`
        return False
    return not any(re.search(rf"\b{kw}\b", low) for kw in _FORBIDDEN)


def _execute_sql(query: str) -> dict:
    """Run a read-only query and return {'data': rows} or {'error': message}.

    Uses a fresh cursor per call so it stays safe if called from multiple threads.
    """
    if not _is_safe(query):
        # WARNING: the model produced a non-read-only query. Benign here (we refuse to run
        # it), but worth surfacing - a spike could signal a prompt-injection attempt.
        logger.warning("sql.execute: rejected unsafe query: %s", snippet(query, 120))
        return {"error": "Only read-only SELECT queries are allowed."}
    try:
        df = _con.cursor().execute(query).df()
    except Exception as exc:
        return {"error": f"SQL error: {exc}"}
    return {"data": df.head(100).to_dict(orient="records")}


# --- LLM: structured SQL generation and judging ------------------------------

class SQLPlan(BaseModel):
    """The planner's structured output: whether it's answerable, why, and the query."""

    answerable: bool = Field(description="True if the question can be answered from the ledger table")
    reasoning: str = Field(description="Brief explanation of how the query answers the question, or why it can't")
    sql: str = Field(default="", description="A single read-only DuckDB SELECT query; empty if not answerable")


class Judgment(BaseModel):
    """The judge's structured verdict on a generated query."""

    is_good: bool = Field(description="True if the SQL correctly and fully answers the question")
    feedback: str = Field(default="", description="If not good, one sentence on how to fix the query")


_llm = get_llm()
_planner = _llm.with_structured_output(SQLPlan)
_judge = _llm.with_structured_output(Judgment)


# --- Graph state -------------------------------------------------------------

class SQLState(TypedDict):
    """State passed between the graph's nodes."""

    question: str
    answerable: bool
    reasoning: str
    sql: str
    data: list
    error: str
    judge_ok: bool
    judge_feedback: str
    answer: str
    grounded: bool
    unsupported: list
    attempts: int


MAX_ATTEMPTS = 2  # how many times the agent may rewrite a query (on error or judge feedback)


# --- Nodes -------------------------------------------------------------------

def _generate(state: SQLState) -> dict:
    """Ask the LLM to write its reasoning and one SQL query for the question."""
    system = SQL_GENERATE_PROMPT.format(
        min_month=_MIN_MONTH, max_month=_MAX_MONTH, max_quarter=_MAX_QUARTER
    )
    human = state["question"]
    if state.get("error"):  # retrying after a failed query
        human += f"\n\nYour previous query failed with:\n{state['error']}\nFix the SQL."
    elif state.get("judge_feedback") and not state.get("judge_ok", True):  # retrying after judge
        human += f"\n\nA reviewer flagged your previous attempt:\n{state['judge_feedback']}\nImprove the query."
    plan = _planner.invoke([SystemMessage(system), HumanMessage(human)])
    attempt = state.get("attempts", 0) + 1
    logger.info("sql.generate: attempt=%s answerable=%s", attempt, plan.answerable)
    logger.debug("sql.generate: sql=%s", snippet(plan.sql, 300))
    return {"answerable": plan.answerable, "reasoning": plan.reasoning,
            "sql": plan.sql, "attempts": attempt}


def _execute(state: SQLState) -> dict:
    """Run the generated SQL against DuckDB."""
    result = _execute_sql(state["sql"])
    if result.get("error"):
        logger.warning("sql.execute: %s", result["error"])
    else:
        logger.info("sql.execute: %s rows", len(result.get("data", [])))
    return {"data": result.get("data", []), "error": result.get("error", "")}


def _judge_node(state: SQLState) -> dict:
    """Optional: have the LLM review whether the query answers the question."""
    human = (f"Question: {state['question']}\n"
             f"Reasoning: {state['reasoning']}\n"
             f"SQL: {state['sql']}\n"
             f"Results (sample): {state['data'][:5]}")
    verdict = _judge.invoke([SystemMessage(SQL_JUDGE_PROMPT), HumanMessage(human)])
    logger.info("sql.judge: is_good=%s %s", verdict.is_good,
                "" if verdict.is_good else snippet(verdict.feedback, 80))
    return {"judge_ok": verdict.is_good, "judge_feedback": verdict.feedback}


def _respond(state: SQLState) -> dict:
    """Phrase the final answer via the shared Responder."""
    if not state.get("answerable", True):
        note = (f"This cannot be answered from the ledger. Reason: {state['reasoning']}. "
                f"Politely tell the user you can't answer it from the available financial data.")
        answer = write_answer(state["question"], note=note)
    elif state.get("error"):
        note = (f"The query failed after retries: {state['error']}. "
                f"Briefly explain that the data could not be retrieved.")
        answer = write_answer(state["question"], note=note)
    else:
        answer = write_answer(state["question"], results=state["data"])
    return {"answer": answer}


def _check(state: SQLState) -> dict:
    """Deterministically verify the answer's numbers came from the data."""
    if state.get("answerable", True) and not state.get("error"):
        result = check_grounding(state["answer"], state["data"], state["question"])
        if not result["grounded"]:
            logger.warning("sql.check: ungrounded numbers in answer: %s", result["unsupported"])
        return {"grounded": result["grounded"], "unsupported": result["unsupported"]}
    return {"grounded": True, "unsupported": []}  # nothing to ground for a decline/error


def _after_generate(state: SQLState) -> str:
    """Skip execution for unanswerable questions."""
    return "execute" if state["answerable"] else "respond"


def _after_execute(state: SQLState) -> str:
    """Retry a failed query, else judge (if enabled), else answer."""
    if state.get("error") and state["attempts"] < MAX_ATTEMPTS:
        return "generate"
    return "judge" if USE_JUDGE else "respond"


def _after_judge(state: SQLState) -> str:
    """Rewrite the query if the judge rejected it and attempts remain, else answer."""
    if not state["judge_ok"] and state["attempts"] < MAX_ATTEMPTS:
        return "generate"
    return "respond"


# --- Build & compile the graph -----------------------------------------------

_graph = StateGraph(SQLState)
_graph.add_node("generate", _generate)
_graph.add_node("execute", _execute)
_graph.add_node("judge", _judge_node)
_graph.add_node("respond", _respond)
_graph.add_node("check", _check)
_graph.add_edge(START, "generate")
_graph.add_conditional_edges("generate", _after_generate,
                             {"execute": "execute", "respond": "respond"})
_graph.add_conditional_edges("execute", _after_execute,
                             {"generate": "generate", "judge": "judge", "respond": "respond"})
_graph.add_conditional_edges("judge", _after_judge,
                             {"generate": "generate", "respond": "respond"})
_graph.add_edge("respond", "check")
_graph.add_edge("check", END)
sql_agent = _graph.compile()


def ask_sql(question: str) -> dict:
    """Answer a question with DuckDB SQL.

    Returns a dict with the agent's `reasoning`, the `sql` it ran, the final `answer`,
    the raw `data` rows, the `grounded` flag with any `unsupported` numbers, and (when
    the judge is enabled) `judge_ok` / `judge_feedback`.
    """
    final = sql_agent.invoke({"question": question})
    return {
        "reasoning": final.get("reasoning", ""),
        "sql": final.get("sql", ""),
        "answer": final.get("answer", ""),
        "data": final.get("data", []),
        "grounded": final.get("grounded", True),
        "unsupported": final.get("unsupported", []),
        "judge_ok": final.get("judge_ok"),
        "judge_feedback": final.get("judge_feedback", ""),
    }
