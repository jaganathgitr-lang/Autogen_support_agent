# AutoGen Customer Support Agents

Two AutoGen (`autogen-agentchat` v0.4+) + Streamlit customer support apps, sharing one environment:

- **`app.py`** -- the Gen AI Architect Program buildathon deliverable: exactly 3 agents in a fixed `RoundRobinGroupChat`, matching the assignment spec precisely.
- **`support_router.py`** -- an extended build: a Manager agent that routes each question to one of 5 department agents, using `SelectorGroupChat` with deterministic routing.

Both use a Serper-backed web search tool and read all API keys from environment variables.

---

## `app.py` -- the buildathon submission

Three `AssistantAgent`s, in a fixed order, one one after another:

```
User question
     |
     v
Assistant              -- answers directly, from its own knowledge. No tools.
     |
     v
Web Search Assistant   -- holds the web-search tool. Searches, then answers
     |                     from the results.
     v
Entry Agent            -- holds the file-writing tool. Saves the query +
     |                     both answers to answers.txt, returns both answers.
     v
Streamlit UI shows both answers live, as each agent finishes its turn
```

| Agent | Tools | Job |
|---|---|---|
| **Assistant** | none | Answers the query directly, from its own knowledge only. |
| **Web Search Assistant** | `web_search` (Serper) | Searches the web for the query, answers from the results. |
| **Entry Agent** | `save_to_file` | Saves the query + both answers to `answers.txt`, returns both to the user. |

### How the sequencing and stopping work

This uses `RoundRobinGroupChat` -- the three agents always speak in the exact order listed, no routing/classification needed since every query goes through all three. The team stops via `MaxMessageTermination(4)`: 1 user message + 3 agent answers = 4, so it halts right after the Entry Agent's turn instead of looping back to the Assistant.

Tool calls (the web search, the file save) inject extra `ToolCallRequestEvent`/`ToolCallExecutionEvent` items into the raw message stream -- confirmed live, a 3-turn run produces 8 total stream items, not 4. `MaxMessageTermination` only counts genuine chat messages toward its limit, not these tool-call events, so it still stops in exactly the right place; the UI's `get_text_answer()` helper filters the stream down to each agent's actual final text answer for display, skipping the tool-call plumbing.

### Run it

```bash
streamlit run app.py
```

---

## `support_router.py` -- extended: department routing

6 agents total (a Manager plus 5 departments):

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

| Agent | Job |
|---|---|
| **Manager** | Reads the question, replies with just one department name: IT, HR, Compliance, Admin, or Account. Never answers the question itself. |
| **IT Agent** | Hardware, software, networks, passwords, system access. |
| **HR Agent** | Leave, benefits, payroll, hiring, workplace policy. |
| **Compliance Agent** | Regulations, data privacy, audits, policy compliance. |
| **Admin Agent** | Office facilities, supplies, travel booking, general admin requests. |
| **Account Agent** | Billing, invoices, subscriptions, account/payment issues. |

Every department agent is explicitly told today's real date, so it searches with the actual current year instead of guessing one from its training data -- confirmed live: asked "what is the end date to submit tax filing?" before this fix, the Compliance agent confidently answered with a 2022/2023 deadline; after the fix, it searches "2026 tax filing deadline" and returns the real, current answer (IRS.gov: April 15, 2026).

### How the routing works

This uses `SelectorGroupChat`, but **not** its default LLM-based speaker selection -- that's a second, independent LLM call guessing who should talk next, which adds a point of failure. Instead, a plain Python `selector_func` reads the Manager's own classification and deterministically routes to the matching agent:

1. First turn → always the Manager.
2. After the Manager replies → parse its one-word department name, hand off to that exact department agent.
3. After the department agent replies → stop (`MaxMessageTermination(3)`: question → Manager's routing → department's answer).

### Run it

```bash
streamlit run support_router.py
```

Type a question, click **Ask**, and watch the live orchestration log show the Manager's decision and the department agent's answer as they happen.

---

## Shared setup

```bash
cd autogen-support-router
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your real keys:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=your-openai-key-here
SERPER_API_KEY=your-serper-key-here
```

- OpenAI key: https://platform.openai.com
- Serper key (free tier): https://serper.dev

Both apps run on `http://localhost:8501` (Streamlit's default) -- don't run them at the same time on the same port.

## Project files

```
autogen-support-router/
├── app.py                # buildathon submission: 3 agents, RoundRobinGroupChat
├── support_router.py     # extended: Manager + 5 department agents, SelectorGroupChat
├── requirements.txt
├── .env.example
├── .env                   # not committed -- your real keys go here
├── .gitignore
└── answers.txt            # generated at runtime by app.py, not committed
```

## Examples

**`app.py`:**
```
Query: How do I reset my password?
Answer 1 (Assistant): [general password-reset steps, from the model's own knowledge]
Answer 2 (Web Search Assistant): [steps grounded in real search results]
-> both saved to answers.txt
```

**`support_router.py`:**
```
Question: My laptop won't connect to the office wifi, what should I do?
Manager's decision: IT
Routed to: IT Agent
Answer: [troubleshooting steps]
```
