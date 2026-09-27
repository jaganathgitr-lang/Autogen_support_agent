# Customer Support Router (AutoGen)

A customer support chatbot built with **AutoGen** (the modern `autogen-agentchat` v0.4+ API) and a **Streamlit** UI. A Manager agent reads the incoming question and routes it to the right department -- that department's agent answers it, grounded in a real web search when the answer needs to be current.

```
User question
     |
     v
Manager        -- classifies the question into exactly one department
     |             (does not answer it itself)
     v
IT / HR / Compliance / Admin / Account agent
     |             -- answers the question, using Serper web search
     |                when the answer needs a current fact or date
     v
Streamlit UI shows the live routing decision, then the answer
```

## The agents

6 agents total (exceeds the "minimum 5" department-agent requirement this was built for):

| Agent | Job |
|---|---|
| **Manager** | Reads the question, replies with just one department name: IT, HR, Compliance, Admin, or Account. Never answers the question itself. |
| **IT Agent** | Hardware, software, networks, passwords, system access. |
| **HR Agent** | Leave, benefits, payroll, hiring, workplace policy. |
| **Compliance Agent** | Regulations, data privacy, audits, policy compliance. |
| **Admin Agent** | Office facilities, supplies, travel booking, general admin requests. |
| **Account Agent** | Billing, invoices, subscriptions, account/payment issues. |

Every department agent has a web search tool backed by Serper (google.serper.dev -- the same search API used in the buildathon-support-crew and linkedin-post-crew projects, for consistency) and is explicitly told today's real date, so it searches with the actual current year instead of guessing one from its training data -- confirmed live: asked "what is the end date to submit tax filing?" before this fix, the Compliance agent confidently answered with a 2022/2023 deadline; after the fix, it searches "2026 tax filing deadline" and returns the real, current answer (IRS.gov: April 15, 2026).

AutoGen's `AssistantAgent` takes plain Python callables as tools (unlike CrewAI's `SerperDevTool` class used in the other two projects), so `web_search()` here calls Serper's REST API directly instead.

## How the routing works

This uses AutoGen's `SelectorGroupChat`, but **not** its default LLM-based speaker selection -- that's a second, independent LLM call guessing who should talk next, which adds a point of failure. Instead, a plain Python `selector_func` reads the Manager's own classification and deterministically routes to the matching agent:

1. First turn → always the Manager.
2. After the Manager replies → parse its one-word department name, hand off to that exact department agent.
3. After the department agent replies → stop (`MaxMessageTermination(3)`: question → Manager's routing → department's answer).

This is more reliable than trusting an LLM selector to infer the right agent from descriptions alone, and it means the Manager's routing decision is always genuinely why a given agent answered -- not a coincidence of a second guess landing on the same agent.

## Live orchestration in the UI

The app uses `team.run_stream()` (not `team.run()`), so the Streamlit UI updates as each agent message arrives, instead of only showing a result once everything finishes:

- A status banner updates the moment the Manager decides ("Manager decided: routing this to **IT**...").
- If the department agent searches the web, the live log shows "*IT Agent is searching the web...*" / "*found some results, reading them...*" instead of dumping the raw tool-call objects.
- The final answer and a "Full conversation" expander (showing the Manager's decision + the department agent's full answer) appear once the team finishes.

## Setup

```bash
cd autogen-support-router
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your real key:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=your-openai-key-here
SERPER_API_KEY=your-serper-key-here
```

- OpenAI key: https://platform.openai.com
- Serper key (free tier): https://serper.dev

## Run it

```bash
streamlit run support_router.py
```

Opens at `http://localhost:8501`. Type a question, click **Ask**, and watch the live orchestration log show the Manager's decision and the department agent's answer as they happen.

## Project files

```
autogen-support-router/
├── support_router.py    # everything: agents, routing, streaming, Streamlit UI
├── requirements.txt
├── .env.example
├── .env                  # not committed -- your real key goes here
└── .gitignore
```

## Example

```
Question: My laptop won't connect to the office wifi, what should I do?
Manager's decision: IT
Routed to: IT Agent
Answer: [troubleshooting steps]
```

```
Question: How many paid leave days do I have left this year?
Manager's decision: HR
Routed to: HR Agent
Answer: [HR-specific guidance]
```
