# app.py
# ---------------------------------------------------------------
# Multi-Agent Customer Support built with AutoGen (AgentChat) + Streamlit.
#
# 3 agents run in a fixed order (RoundRobinGroupChat):
#   1. Assistant             -> answers the query directly, from its own
#                                knowledge only. No tools.
#   2. Web Search Assistant  -> holds the web-search tool (Serper). Searches
#                                the web for the query and answers from the
#                                results.
#   3. Entry Agent           -> holds the file-writing tool. Saves the query
#                                and both prior answers to answers.txt, then
#                                returns both answers to the user.
#
# The team stops right after the Entry Agent's turn via
# MaxMessageTermination(4) -- 1 user message + 3 agent answers -- so it
# never loops back to the Assistant. (Tool-call plumbing events, e.g. from
# the web-search or file-save calls, don't count toward this: AutoGen's
# MaxMessageTermination only counts real chat messages, confirmed live
# while building this.)
# ---------------------------------------------------------------
#
# SETUP (run once in the terminal):
#   pip install -r requirements.txt
#
#   Put your keys in a .env file next to this script:
#     OPENAI_API_KEY=your-openai-key
#     SERPER_API_KEY=your-serper-key      (from https://serper.dev - used for web search)
#
# RUN:
#   streamlit run app.py
# ---------------------------------------------------------------

import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage, ToolCallExecutionEvent, ToolCallRequestEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

ANSWERS_FILE = "answers.txt"


# ---------------------------------------------------------------------------
# Tools -- plain Python functions, given to exactly one agent each.
# ---------------------------------------------------------------------------
def web_search(query: str) -> str:
    """Search the web for the query and return the top results as text."""
    import requests

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Web search is unavailable (SERPER_API_KEY is not set)."

    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query},
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("organic", [])[:3]
    if not results:
        return "No results found."
    return "\n\n".join(f"{r.get('title', '')}\n{r.get('snippet', '')}" for r in results)


def save_to_file(query: str, answer_1: str, answer_2: str) -> str:
    """Save the customer's query and both agent answers to answers.txt.
    Pass the exact query, answer_1 (the Assistant's answer), and answer_2
    (the Web Search Assistant's answer) -- do not summarize or reword them."""
    with open(ANSWERS_FILE, "a", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        f.write(f"Answer 1 (Assistant):\n{answer_1}\n\n")
        f.write(f"Answer 2 (Web Search Assistant):\n{answer_2}\n\n")
        f.write("-" * 60 + "\n\n")
    return f"Saved the query and both answers to {ANSWERS_FILE}."


# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------
def build_team():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    assistant = AssistantAgent(
        name="Assistant",
        model_client=model_client,
        system_message=(
            "You are the Assistant. Answer the user's query directly and clearly, using only "
            "your own knowledge -- you have no tools. Do not mention that you lack tools; just "
            "answer helpfully."
        ),
    )

    web_search_assistant = AssistantAgent(
        name="Web_Search_Assistant",
        model_client=model_client,
        tools=[web_search],
        reflect_on_tool_use=True,
        system_message=(
            "You are the Web Search Assistant. Always call the web_search tool with the "
            "user's original query before answering -- never answer from memory alone. "
            "Then write a clear answer based on what the search returned."
        ),
    )

    entry_agent = AssistantAgent(
        name="Entry_Agent",
        model_client=model_client,
        tools=[save_to_file],
        reflect_on_tool_use=True,
        system_message=(
            "You are the Entry Agent. The conversation above contains the user's original "
            "query, the Assistant's answer, and the Web Search Assistant's answer. Call the "
            "save_to_file tool exactly ONCE, passing the exact query, answer_1 (the "
            "Assistant's answer), and answer_2 (the Web Search Assistant's answer) -- do not "
            "summarize, shorten, or reword either answer, and do not call the tool more than "
            "once. After saving, reply with both answers, clearly labeled 'Answer 1' and "
            "'Answer 2'."
        ),
    )

    team = RoundRobinGroupChat(
        participants=[assistant, web_search_assistant, entry_agent],
        # 1 user message + 3 agent answers = 4 -- stops right after the Entry
        # Agent's turn, before the round-robin would loop back to Assistant.
        termination_condition=MaxMessageTermination(4),
    )
    return team


async def run_support_team_stream(query: str, on_message):
    """Runs the team via run_stream() so the UI can render each agent's turn
    live, instead of only showing a result once everything finishes."""
    team = build_team()
    final_result = None
    async for item in team.run_stream(task=query):
        if isinstance(item, TaskResult):
            final_result = item
        else:
            on_message(item)
    return final_result


def get_text_answer(messages, source_name):
    """Finds the given agent's actual text answer, skipping over any
    ToolCallRequestEvent/ToolCallExecutionEvent plumbing from its turn."""
    for m in reversed(messages):
        if isinstance(m, TextMessage) and m.source == source_name:
            return m.content
    return None


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Customer Support (AutoGen)", page_icon="🎧", layout="wide")
st.title("🎧 Customer Support Team (AutoGen)")
st.write(
    "Ask a question. Three agents run in order: the Assistant answers directly, "
    "the Web Search Assistant checks the web, and the Entry Agent saves both "
    "answers to answers.txt and shows them to you."
)

query = st.text_area("Your question or task", placeholder="Example: How do I reset my password?")

if st.button("Ask") and query.strip():
    st.markdown("### Live conversation")
    status_placeholder = st.empty()
    log_container = st.container()
    status_placeholder.info("Assistant is answering...")

    def render_message(m):
        pretty_name = m.source.replace("_", " ")
        if isinstance(m, ToolCallRequestEvent):
            st.markdown(f"*{pretty_name} is using a tool...*")
        elif isinstance(m, ToolCallExecutionEvent):
            st.markdown(f"*{pretty_name} got a tool result, reading it...*")
        elif isinstance(m, TextMessage):
            st.markdown(f"**{pretty_name}:** {m.content}")

    def on_message(m):
        if m.source == "user":
            return  # already shown in the input box
        if isinstance(m, TextMessage):
            if m.source == "Assistant":
                status_placeholder.info("Web Search Assistant is searching...")
            elif m.source == "Web_Search_Assistant":
                status_placeholder.info("Entry Agent is saving the results...")
        with log_container:
            render_message(m)

    with st.spinner("Working..."):
        final_result = asyncio.run(run_support_team_stream(query.strip(), on_message))

    messages = final_result.messages if final_result else []
    answer_1 = get_text_answer(messages, "Assistant")
    answer_2 = get_text_answer(messages, "Web_Search_Assistant")

    if not messages or answer_1 is None or answer_2 is None:
        status_placeholder.error("Something went wrong -- the team stopped before both answers were produced.")
    else:
        status_placeholder.success(f"Done -- saved to {ANSWERS_FILE}")

        st.subheader("Answer 1 -- Assistant")
        st.markdown(answer_1)

        st.subheader("Answer 2 -- Web Search Assistant")
        st.markdown(answer_2)

        with st.expander("Full conversation"):
            for m in messages:
                if m.source == "user":
                    st.markdown(f"**User:** {m.content}")
                else:
                    render_message(m)
