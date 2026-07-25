"""Text-to-SQL analyst agent (DuckDB), built as an explicit LangGraph pipeline.

Flow:  generate -> execute -> respond
- generate: the LLM writes its reasoning and ONE read-only SQL query (structured output).
- execute:  DuckDB runs the query; on a SQL error the graph loops back to generate so
            the agent can self-correct (up to MAX_ATTEMPTS).
- respond:  the LLM phrases the final answer from the returned rows.

All arithmetic happens in SQL — the model never computes numbers itself. The agent
returns a structured result: {reasoning, sql, answer, data}.

Data access: the property-ledger parquet is exposed as a read-only DuckDB view named
`ledger` (in-memory; the parquet file is never modified).
"""

import re
from typing import TypedDict

import duckdb
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..config import DATA_PATH, get_llm
from ..prompts import SQL_ANSWER_PROMPT, SQL_GENERATE_PROMPT

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
        return {"error": "Only read-only SELECT queries are allowed."}
    try:
        df = _con.cursor().execute(query).df()
    except Exception as exc:
        return {"error": f"SQL error: {exc}"}
    return {"data": df.head(100).to_dict(orient="records")}


# --- LLM: structured SQL generation ------------------------------------------

class SQLPlan(BaseModel):
    """The planner's structured output: whether it's answerable, why, and the query."""

    answerable: bool = Field(description="True if the question can be answered from the ledger table")
    reasoning: str = Field(description="Brief explanation of how the query answers the question, or why it can't")
    sql: str = Field(default="", description="A single read-only DuckDB SELECT query; empty if not answerable")


_llm = get_llm()
_planner = _llm.with_structured_output(SQLPlan)


# --- Graph state -------------------------------------------------------------

class SQLState(TypedDict):
    """State passed between the graph's nodes."""

    question: str
    answerable: bool
    reasoning: str
    sql: str
    data: list
    error: str
    answer: str
    attempts: int


MAX_ATTEMPTS = 2  # how many times the agent may rewrite a failing query


# --- Nodes -------------------------------------------------------------------

def _generate(state: SQLState) -> dict:
    """Ask the LLM to write its reasoning and one SQL query for the question."""
    system = SQL_GENERATE_PROMPT.format(
        min_month=_MIN_MONTH, max_month=_MAX_MONTH, max_quarter=_MAX_QUARTER
    )
    human = state["question"]
    if state.get("error"):  # we're retrying after a failed query
        human += f"\n\nYour previous query failed with:\n{state['error']}\nFix the SQL."
    plan = _planner.invoke([SystemMessage(system), HumanMessage(human)])
    return {"answerable": plan.answerable, "reasoning": plan.reasoning,
            "sql": plan.sql, "attempts": state.get("attempts", 0) + 1}


def _execute(state: SQLState) -> dict:
    """Run the generated SQL against DuckDB."""
    result = _execute_sql(state["sql"])
    return {"data": result.get("data", []), "error": result.get("error", "")}


def _respond(state: SQLState) -> dict:
    """Phrase the final answer from the results (or explain why there are none)."""
    if not state.get("answerable", True):
        human = (f"Question: {state['question']}\n"
                 f"This cannot be answered from the ledger. Reason: {state['reasoning']}\n"
                 f"Politely tell the user you can't answer it from the available financial data.")
    elif state.get("error"):
        human = (f"Question: {state['question']}\n"
                 f"The query failed after retries: {state['error']}\n"
                 f"Briefly explain that the data could not be retrieved.")
    else:
        human = (f"Question: {state['question']}\n"
                 f"SQL: {state['sql']}\nResults: {state['data']}")
    answer = _llm.invoke([SystemMessage(SQL_ANSWER_PROMPT), HumanMessage(human)])
    return {"answer": answer.content}


def _after_generate(state: SQLState) -> str:
    """Skip execution for unanswerable questions."""
    return "execute" if state["answerable"] else "respond"


def _after_execute(state: SQLState) -> str:
    """Retry a failed query up to MAX_ATTEMPTS, otherwise answer."""
    if state.get("error") and state["attempts"] < MAX_ATTEMPTS:
        return "generate"
    return "respond"


# --- Build & compile the graph -----------------------------------------------

_graph = StateGraph(SQLState)
_graph.add_node("generate", _generate)
_graph.add_node("execute", _execute)
_graph.add_node("respond", _respond)
_graph.add_edge(START, "generate")
_graph.add_conditional_edges("generate", _after_generate,
                             {"execute": "execute", "respond": "respond"})
_graph.add_conditional_edges("execute", _after_execute,
                             {"generate": "generate", "respond": "respond"})
_graph.add_edge("respond", END)
sql_agent = _graph.compile()


def ask_sql(question: str) -> dict:
    """Answer a question with DuckDB SQL.

    Returns a dict with the agent's `reasoning`, the `sql` it ran, the final `answer`,
    and the raw `data` rows.
    """
    final = sql_agent.invoke({"question": question})
    return {
        "reasoning": final.get("reasoning", ""),
        "sql": final.get("sql", ""),
        "answer": final.get("answer", ""),
        "data": final.get("data", []),
    }
