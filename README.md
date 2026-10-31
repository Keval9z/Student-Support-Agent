# Intelligent & Safe Student Support Agent

One application demonstrates both assignments:
- Assignment 1: baseline agent, tools, handbook retrieval, and persistent non-sensitive memory.
- Assignment 2: the same agent extended with login, own-record authorization, staff approval, and audit logging.

## Setup (Windows / PyCharm)
1. Open the generated project folder in PyCharm.
2. In the terminal run `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env`.
4. Add your Groq API key. Add a LangSmith API key if you want cloud tracing.
5. Run `python evaluation.py`.
6. Run `streamlit run app.py`.

## Model and tracing
The default model is `openai/gpt-oss-120b`, accessed through Groq with
`ChatGroq` from `langchain-groq`. Use a model currently available to your
Groq account that supports tool calling.

With `LANGSMITH_TRACING=true` and a valid `LANGSMITH_API_KEY`, LangChain
runs can be traced in the LangSmith project named in `.env`. The application
also writes a local summary to `logs/traces.jsonl`. The local `audit_log.json`
is separate and records authorization and approval decisions.

Do not place passwords, OTPs, API keys, or other secrets in prompts, memory,
support tickets, or trace metadata.

## Demo accounts
Student IDs: `STU1001`, `STU1002`, `STU1003`
Staff ID: `ADMIN`
Demo student password: `Student123!`
Demo staff password: `Admin123!`

These are classroom-only credentials and fictional records.

## Limitations
- Handbook retrieval is keyword-based, not a production semantic vector database.
- Records and runtime state are stored in local JSON files.
- Demo password hashing and local storage are not production-grade.
- Use fictional data only. Real deployment requires secure password hashing,
  a database, robust session management, access controls, encryption, and
  institutional security review.
