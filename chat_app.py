"""Streamlit chat demo for the Supply Chain Intelligence assistant."""
import os
import uuid

import streamlit as st

try:                                   # on Streamlit Cloud the key comes from Secrets
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from src.agent.chatbot import build_graph

PROMPT_VERSION = "v2"
MAX_QUESTIONS = 15

st.set_page_config(page_title="Supply Chain Assistant", page_icon="📦")
st.title("Supply Chain Intelligence Assistant")
st.caption("Ask about product inventory policies, the replenishment simulation, or inventory concepts. "
           "Every number comes from a tool, shown under each answer.")

with st.sidebar:
    st.subheader("Try asking")
    st.markdown("- What is the reorder point and EOQ for product 365?\n"
                "- What does the policy look like for the Perfect Rip Deck?\n"
                "- Compare the replenishment policies for product 957\n"
                "- What is an (s, Q) policy?\n"
                "- Days of supply with 1150 units on hand and 290 sold per day?")
    st.subheader("About")
    st.markdown("Built on the DataCo supply-chain dataset: forecasting, inventory policies, a RAG knowledge base "
                "and a LangGraph agent with LangChain tools. The policy comparison runs a simulation that "
                "**assumes** a lost-sale cost of 25% of unit price (not part of the original dataset).")
    st.caption(f"Prompt {PROMPT_VERSION} | limit {MAX_QUESTIONS} questions per session")


@st.cache_resource
def get_graph():
    return build_graph(PROMPT_VERSION)


def tool_trace(msgs):
    start = max(i for i, m in enumerate(msgs) if m.type == "human")
    new = msgs[start + 1:]
    results = {m.tool_call_id: m.content for m in new if m.type == "tool"}
    trace = []
    for m in new:
        for c in getattr(m, "tool_calls", None) or []:
            trace.append({"name": c["name"], "args": c["args"], "result": str(results.get(c["id"], ""))[:600]})
    return trace


def show(role, text, trace):
    with st.chat_message(role):
        st.markdown(text.replace("$", "\\$"))
        if trace:
            with st.expander(f"Tools used ({len(trace)})"):
                for t in trace:
                    st.code(f"{t['name']}({t['args']})\n-> {t['result']}")


if "tid" not in st.session_state:
    st.session_state.tid = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.count = 0

for role, text, trace in st.session_state.history:
    show(role, text, trace)

q = st.chat_input("Ask a question, e.g. 'Compare the policies for product 365'")
if q:
    if st.session_state.count >= MAX_QUESTIONS:
        st.warning("Question limit reached for this session. Refresh the page to start again.")
    else:
        st.session_state.count += 1
        show("user", q, None)
        st.session_state.history.append(("user", q, None))
        try:
            with st.spinner("Thinking..."):
                msgs = get_graph().invoke({"messages": [("user", q)]},
                                          {"configurable": {"thread_id": st.session_state.tid}})["messages"]
            answer, trace = msgs[-1].content, tool_trace(msgs)
        except Exception as ex:
            answer, trace = f"Sorry, the model call failed ({type(ex).__name__}). Please try again in a minute.", None
        show("assistant", answer, trace)
        st.session_state.history.append(("assistant", answer, trace))
