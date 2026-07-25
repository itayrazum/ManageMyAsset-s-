"""Streamlit chat UI for the property asset-management assistant.

Streams the LangGraph pipeline node-by-node so the user can watch where the request is
routed (analytics / clarify / out-of-scope / blocked) and how the answer is produced
(SQL, grounding, cache). Each session gets its own thread_id, so the assistant remembers
the conversation.
"""

import os
import uuid

import pandas as pd
import streamlit as st

import theme

# On Streamlit Community Cloud there is no .env file — secrets come from st.secrets. Copy
# them into the environment BEFORE importing src, so config.py's os.getenv(...) finds the
# API key. Locally this is a harmless no-op (the .env file is used instead).
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

from langchain_core.messages import HumanMessage

from src.graph import assistant

st.set_page_config(page_title="ManageMyAsset(s)", page_icon="🏢", layout="centered")

# Fail fast with a clear message if the API key never made it into the environment.
from src.config import ANTHROPIC_API_KEY, LLM_PROVIDER, OPENAI_API_KEY

if not (ANTHROPIC_API_KEY if LLM_PROVIDER == "anthropic" else OPENAI_API_KEY):
    st.error(
        "No API key found. Add it under **Manage app → Settings → Secrets** in TOML form:\n\n"
        '```toml\nCLAUDE_API = "sk-ant-..."\nLLM_PROVIDER = "anthropic"\nANTHROPIC_MODEL = "claude-haiku-4-5"\n```'
    )
    st.stop()

# Apply the custom dark theme.
st.markdown(theme.theme_css(), unsafe_allow_html=True)

INTENT_BADGE = {
    "analytics": "🔍 Analytics",
    "visualize": "📊 Visualization",
    "clarify": "❓ Clarifying",
    "out_of_scope": "↪️ Out of scope",
    "blocked": "🚫 Blocked",
}

EXAMPLES = [
    "What is the total P&L for all properties in 2024?",
    "Which building is most profitable, and by how much?",
    "Show me revenue by month in 2025",
    "How does Q1 2025 compare to Q1 2024?",
]

AVATARS = {"user": "🧑‍💼", "assistant": "📊"}

# --- Session state -----------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content, trace?}


def render_chart(trace: dict) -> None:
    """Render the chart for a visualization answer (always shown — it *is* the answer)."""
    if not trace.get("chart_type") or not trace.get("chart_data"):
        return
    df = pd.DataFrame(trace["chart_data"])
    x, y = trace["chart_x"], trace["chart_y"]
    if trace["chart_type"] == "line":
        st.line_chart(df, x=x, y=y)
    else:
        st.bar_chart(df, x=x, y=y)


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
                elif node == "visualize":
                    st.write("🗄️ Queried the ledger")
                    st.write("📊 Built a chart from the data")
                elif node == "clarify":
                    st.write("❓ Need a bit more detail")
                elif node in ("out_of_scope", "blocked"):
                    st.write("🛡️ Declining — outside what I can answer")
        label = f"Path: {' → '.join(trace['path'])}" if test_mode else "Done"
        status.update(label=label, state="complete", expanded=False)

    st.markdown(trace.get("answer", ""))
    render_chart(trace)
    render_trace(trace)
    return trace


def handle(prompt: str) -> None:
    """Process a new user prompt: show it, run the assistant, store both."""
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
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
    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.toggle("🔬 Test mode", value=True, key="test_mode",
              help="Show the agent's reasoning, routing path, and SQL for each answer.")

# --- Main --------------------------------------------------------------------
st.markdown('<div class="hero"><span class="logo">🏢</span>'
            '<span class="grad">ManageMyAsset(s)</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A multi-agent assistant over your property ledger. '
            'Toggle <b>Test mode</b> in the sidebar to see how each answer is produced.</div>',
            unsafe_allow_html=True)

for message in st.session_state.history:
    with st.chat_message(message["role"], avatar=AVATARS.get(message["role"])):
        st.markdown(message["content"])
        if message.get("trace"):
            render_chart(message["trace"])
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
