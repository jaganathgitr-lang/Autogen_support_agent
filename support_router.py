# support_router.py
# ---------------------------------------------------------------
# A Customer Support Router built with AutoGen + Streamlit.
#
# 6 agents work together:
#   1. Manager      -> reads the question and decides which department should answer it
#   2. IT Agent          -> answers IT questions
#   3. HR Agent           -> answers HR questions
#   4. Compliance Agent   -> answers compliance questions
#   5. Admin Agent        -> answers admin questions
#   6. Account Agent      -> answers account/billing questions
#
# The Manager doesn't answer the question itself -- it only decides which
# department agent should. A Python routing function (not another LLM call)
# reads the Manager's decision and hands the conversation to the matching
# department agent, so routing is deterministic and never mis-fires.
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
#   streamlit run support_router.py
# ---------------------------------------------------------------

import asyncio
import os
from datetime import date

import streamlit as st
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage, ToolCallExecutionEvent, ToolCallRequestEvent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

DEPARTMENTS = ["IT", "HR", "Compliance", "Admin", "Account"]

# A web search tool backed by Serper (google.serper.dev) -- the same search
# API used in buildathon-support-crew and linkedin-post-crew, for consistency
# across projects, and generally more reliable/structured results than the
# free DuckDuckGo library this started with. AutoGen's AssistantAgent takes
# plain Python callables as tools (unlike CrewAI's SerperDevTool class), so
# this calls Serper's REST API directly.
#
# Without this, every department agent only has what GPT-4o-mini learned
# during training, which goes stale fast for anything date-specific (tax
# deadlines, current policy changes, current pricing, etc.) -- confirmed
# live: asked "what is the end date to submit tax filing?" and the Compliance
# agent confidently answered with a 2022/2023 tax-year deadline instead of
# admitting it didn't know the current one. Every agent below gets this tool
# so it can look up a real, current answer instead of guessing from memory.
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


_SEARCH_INSTRUCTION = (
    f" Today's actual date is {date.today().isoformat()} -- your training data "
    "is older than that, so don't assume the current year is whatever year "
    "feels 'current' from what you remember. If the question needs a current "
    "fact, date, or figure you're not certain is still accurate (deadlines, "
    "policy changes, prices, current events), use the web_search tool to check "
    "before answering, and phrase your search using the actual current date "
    "above (e.g. the real current year), not a guessed one. If you search and "
    "still can't find a definite answer, say so honestly instead of guessing."
)

DEPARTMENT_INFO = {
    "IT": (
        "IT_Agent",
        "You are the IT support agent. Answer the user's question about hardware, "
        "software, networks, passwords, or system access -- clearly, helpfully, and "
        "with concrete next steps." + _SEARCH_INSTRUCTION,
    ),
    "HR": (
        "HR_Agent",
        "You are the HR support agent. Answer the user's question about leave, "
        "benefits, payroll, hiring, or workplace policy -- clearly and helpfully."
        + _SEARCH_INSTRUCTION,
    ),
    "Compliance": (
        "Compliance_Agent",
        "You are the Compliance support agent. Answer the user's question about "
        "regulations, data privacy, audits, or policy compliance -- clearly and "
        "carefully, noting when something needs a human compliance officer's "
        "sign-off." + _SEARCH_INSTRUCTION,
    ),
    "Admin": (
        "Admin_Agent",
        "You are the Admin support agent. Answer the user's question about office "
        "facilities, supplies, travel booking, or general administrative requests "
        "-- clearly and helpfully." + _SEARCH_INSTRUCTION,
    ),
    "Account": (
        "Account_Agent",
        "You are the Account support agent. Answer the user's question about "
        "billing, invoices, subscriptions, or account/payment issues -- clearly "
        "and helpfully." + _SEARCH_INSTRUCTION,
    ),
}


def build_team():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    manager = AssistantAgent(
        name="Manager",
        model_client=model_client,
        system_message=(
            "You are a customer support manager. Read the user's question and decide "
            "which ONE department should answer it: IT, HR, Compliance, Admin, or "
            "Account. Reply with ONLY the department name -- no explanation."
        ),
    )

    dept_agents = {
        dept: AssistantAgent(
            name=name,
            model_client=model_client,
            system_message=system_message,
            tools=[web_search],
            reflect_on_tool_use=True,
        )
        for dept, (name, system_message) in DEPARTMENT_INFO.items()
    }

    def selector_func(messages):
        last = messages[-1]
        if last.source == "user":
            return "Manager"
        if last.source == "Manager":
            content = last.content.strip().lower()
            for dept, agent in dept_agents.items():
                if dept.lower() in content:
                    return agent.name
            return dept_agents["Admin"].name  # fallback if the Manager's reply is unclear
        return None  # the department agent has answered -- let termination stop the team

    team = SelectorGroupChat(
        participants=[manager, *dept_agents.values()],
        model_client=model_client,
        selector_func=selector_func,
        termination_condition=MaxMessageTermination(3),  # user question -> Manager's routing -> department answer
    )
    return team


async def run_support_team_stream(question: str, on_message):
    """Runs the team via run_stream() instead of run(), calling on_message(msg)
    for each agent message AS IT ARRIVES -- this is what makes the Manager's
    routing decision (and then the department agent's answer) visible live in
    the UI, instead of only after the whole team finishes. run_stream() yields
    each BaseChatMessage as it's produced, then a final TaskResult summarizing
    the whole run -- same pattern the course's own Console() helper uses."""
    team = build_team()
    final_result = None
    async for item in team.run_stream(task=question):
        if isinstance(item, TaskResult):
            final_result = item
        else:
            on_message(item)
    return final_result


# ---------- The webpage ----------
st.set_page_config(page_title="Customer Support Router", page_icon="🎧", layout="wide")
st.title("🎧 Customer Support Router (AutoGen)")
st.write(
    "Ask a question. A Manager agent reads it and routes it to the right department "
    f"({', '.join(DEPARTMENTS)}) -- that department's agent answers it."
)

question = st.text_area(
    "Your question",
    placeholder="Example: My laptop won't connect to the office wifi, what should I do?",
)

if st.button("Ask") and question.strip():
    st.markdown("### Live orchestration")
    status_placeholder = st.empty()
    log_container = st.container()
    status_placeholder.info("Manager is reading your question...")

    def render_message(m):
        """Render one streamed item in the live log. A department agent's turn
        can include tool-call plumbing (ToolCallRequestEvent/ToolCallExecutionEvent)
        before its real answer -- those aren't plain text (they're raw
        FunctionCall/FunctionExecutionResult objects), so show a friendly
        'searching...' line for them instead of dumping the raw repr."""
        pretty_name = m.source.replace("_", " ")
        if isinstance(m, ToolCallRequestEvent):
            st.markdown(f"*{pretty_name} is searching the web...*")
        elif isinstance(m, ToolCallExecutionEvent):
            st.markdown(f"*{pretty_name} found some results, reading them...*")
        elif isinstance(m, TextMessage):
            st.markdown(f"**{pretty_name}:** {m.content}")
        # else: an event type we don't specifically handle -- skip it in the UI

    def on_message(m):
        if m.source == "user":
            return  # already shown in the input box, no need to echo it
        if isinstance(m, TextMessage) and m.source == "Manager":
            status_placeholder.info(f"Manager decided: routing this to **{m.content.strip()}**...")
        with log_container:
            render_message(m)

    with st.spinner("Working..."):
        final_result = asyncio.run(run_support_team_stream(question.strip(), on_message))

    messages = final_result.messages if final_result else []
    manager_msg = next((m for m in messages if isinstance(m, TextMessage) and m.source == "Manager"), None)
    # Termination always fires right after the department agent's final text
    # reply is added (confirmed live: MaxMessageTermination counts only real
    # chat messages, not tool-call events, so those never delay it) -- so the
    # last message is reliably that final answer, however many tool calls
    # happened before it.
    answer_msg = messages[-1] if messages else None

    if not messages or manager_msg is None or answer_msg is None or answer_msg.source in ("user", "Manager"):
        status_placeholder.error("Something went wrong -- the team stopped before a department agent could answer.")
    else:
        status_placeholder.success(f"Routed to: **{answer_msg.source.replace('_', ' ')}**")

        st.subheader("Answer")
        st.markdown(answer_msg.content)

        with st.expander("Full conversation (Manager + department agent)"):
            for m in messages:
                if m.source == "user":
                    st.markdown(f"**User:** {m.content}")
                else:
                    render_message(m)
