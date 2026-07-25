"""The Responder: turns already-computed results into the final natural-language answer.

Kept separate from the agents that *produce* the numbers so it can be shared by every
lane (analytics, clarify, decline) in the full graph. It runs a little warmer than the
analytical steps for more natural phrasing, but is strictly instructed to report only
numbers it is given — never to compute anything itself.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import RESPONDER_TEMPERATURE, get_llm
from ..prompts import CHART_CAPTION_PROMPT, RESPONDER_PROMPT

_llm = get_llm(temperature=RESPONDER_TEMPERATURE)


def write_answer(question: str, results=None, note: str = "") -> str:
    """Write the final answer for a question.

    Pass `results` (rows from a query) for a normal answer, or `note` to describe a
    decline/error case (e.g. the question is out of scope or the query failed).
    """
    if note:
        human = f"Question: {question}\n{note}"
    else:
        human = f"Question: {question}\nResults: {results}"
    return _llm.invoke([SystemMessage(RESPONDER_PROMPT), HumanMessage(human)]).content


def write_caption(question: str, rows: list) -> str:
    """Write a one-sentence caption for a chart from its aggregated data points."""
    human = f"Question: {question}\nChart data: {rows}"
    return _llm.invoke([SystemMessage(CHART_CAPTION_PROMPT), HumanMessage(human)]).content
