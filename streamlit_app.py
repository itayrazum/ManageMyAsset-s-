"""Streamlit chat UI for the property asset-management assistant.

Streams the LangGraph pipeline node-by-node so the user can watch where the request is
routed (analytics / clarify / out-of-scope / blocked) and how the answer is produced
(SQL, grounding, cache). Each session gets its own thread_id, so the assistant remembers
the conversation.
"""

import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from src.graph import assistant

st.set_page_config(page_title="ManageMyAsset(s)", page_icon="🏢", layout="centered")

INTENT_BADGE = {
    "analytics": "🔍 Analytics",
    "clarify": "❓ Clarifying",
    "out_of_scope": "↪️ Out of scope",
    "blocked": "🚫 Blocked",
}

EXAMPLES = [
    "What is the total P&L for all properties in 2024?",
    "Which building is most profitable, and by how much?",
    "Who are my top tenants, and is anything unusual?",
    "How does Q1 2025 compare to Q1 2024?",
]

# --- Session state -----------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content, trace?}


def render_trace(trace: dict) -> None:
    """Render the collapsible 'how I got this' panel — only in test mode."""
    if not st.session_state.get("test_mode", True):
        return
    with st.expander("🧠 Agent trace"):
        st.markdown(f"**Path:** `{' → '.join(trace.get('path', []))}`")
        st.markdown(f"**Intent:** {INTENT_BADGE.get(trace.get('intent'), trace.get('intent'))}")
        if trace.get("route_reason"):
            st.caption(trace["route_reason"])
        if trace.get("standalone_question"):
            st.markdown(f"**Resolved question:** {trace['standalone_question']}")
        if trace.get("reasoning"):
            st.markdown(f"**Reasoning:** {trace['reasoning']}")
        if trace.get("sql"):
            st.code(trace["sql"], language="sql")
        if trace.get("intent") == "analytics":
            bits = []
            bits.append("✅ Grounded" if trace.get("grounded") else "⚠️ Not fully grounded")
            bits.append("⚡ From cache" if trace.get("cached") else "🧮 Freshly computed")
            st.caption("  •  ".join(bits))


def run_assistant(prompt: str) -> dict:
    """Stream the graph for a prompt, narrating the path live; return the trace."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    test_mode = st.session_state.get("test_mode", True)
    trace: dict = {"path": []}

    with st.status("Thinking…", expanded=test_mode) as status:
        for update in assistant.stream({"messages": [HumanMessage(prompt)]}, config,
                                       stream_mode="updates"):
            for node, delta in update.items():
                trace["path"].append(node)
                trace.update({k: v for k, v in delta.items() if k != "messages"})
                if not test_mode:
                    continue  # in normal mode, don't narrate the internal steps
                if node == "route":
                    st.write(f"🧭 Routed to **{INTENT_BADGE.get(trace.get('intent'), trace.get('intent'))}**")
                    if trace.get("route_reason"):
                        st.caption(trace["route_reason"])
                elif node == "analytics":
                    if trace.get("sql"):
                        st.write("🗄️ Wrote and ran SQL over the ledger")
                    st.write("✅ Answer grounded in the data" if trace.get("grounded")
                             else "⚠️ Some figures could not be grounded")
                elif node == "clarify":
                    st.write("❓ Need a bit more detail")
                elif node in ("out_of_scope", "blocked"):
                    st.write("🛡️ Declining — outside what I can answer")
        label = f"Path: {' → '.join(trace['path'])}" if test_mode else "Done"
        status.update(label=label, state="complete", expanded=False)

    st.markdown(trace.get("answer", ""))
    render_trace(trace)
    return trace


def handle(prompt: str) -> None:
    """Process a new user prompt: show it, run the assistant, store both."""
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            trace = run_assistant(prompt)
            st.session_state.history.append(
                {"role": "assistant", "content": trace.get("answer", ""), "trace": trace})
        except Exception as exc:  # keep the app alive on an API/LLM hiccup
            msg = f"Sorry — something went wrong: {exc}"
            st.error(msg)
            st.session_state.history.append({"role": "assistant", "content": msg})


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("🏢 ManageMyAsset(s)")
    st.markdown(
        "Ask about your **property financial ledger** — P&L, revenue, expenses, tenants, "
        "properties, and time comparisons.\n\n"
        "**Pipeline:** router → analytics (SQL) / clarify / decline. Every figure is "
        "computed in SQL and grounding-checked; repeats are cached."
    )
    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.toggle("🔬 Test mode", value=True, key="test_mode",
              help="Show the agent's reasoning, routing path, and SQL for each answer.")

# --- Main --------------------------------------------------------------------
st.title("🏢 ManageMyAsset(s)")
st.caption("A multi-agent assistant over your property ledger. Expand **Agent trace** to see how each answer is produced.")

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("trace"):
            render_trace(message["trace"])

# Welcome examples (only before the first message)
if not st.session_state.history:
    st.markdown("#### Try one of these:")
    for example in EXAMPLES:
        if st.button(example, key=f"ex_{example}", use_container_width=True):
            st.session_state.pending = example
            st.rerun()

prompt = st.chat_input("Ask about your portfolio…")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")
if prompt:
    handle(prompt)
    st.rerun()  # redraw from history so the welcome examples clear and the trace persists
