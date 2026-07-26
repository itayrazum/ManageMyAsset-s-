"""Tests for the router's history sanitization (src/agents/router.py). No LLM calls."""

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.router import _recent


def test_drops_empty_and_whitespace_messages():
    # A blank input the guard already handled (and any empty lane result) must be dropped,
    # so it never reaches the API as an empty text block (Anthropic 400s on those).
    msgs = [HumanMessage("   "), AIMessage(""), HumanMessage("what is my p&l in 2024?")]
    kept = _recent(msgs)
    assert [m.content for m in kept] == ["what is my p&l in 2024?"]


def test_result_starts_on_a_user_turn():
    msgs = [AIMessage("leftover"), HumanMessage("hello")]
    assert isinstance(_recent(msgs)[0], HumanMessage)


def test_keeps_normal_conversation():
    msgs = [HumanMessage("hi"), AIMessage("hello there"), HumanMessage("bye")]
    assert len(_recent(msgs)) == 3
