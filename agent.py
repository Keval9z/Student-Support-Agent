import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

from tools import (
    TOOL_SCHEMAS, get_fee_status, get_scholarship_status,
    search_student_policy, create_support_request, request_programme_change
)
from memory import remember_student_fact, recall_student_memory

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TRACE_LOG_PATH = Path("logs/traces.jsonl")


def build_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured in .env")
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY,
    )


def execute_tool(tool_name: str, args: Dict[str, Any],
                 session_user_id: str, secure_mode: bool) -> Dict[str, Any]:
    """Run a tool using the identity supplied by the authenticated app session."""
    if tool_name == "get_fee_status":
        return get_fee_status(
            args.get("student_id", session_user_id), session_user_id, secure_mode
        )
    if tool_name == "get_scholarship_status":
        return get_scholarship_status(
            args.get("student_id", session_user_id), session_user_id, secure_mode
        )
    if tool_name == "search_student_policy":
        return search_student_policy(args.get("query", ""))
    if tool_name == "remember_student_fact":
        return remember_student_fact(session_user_id, args.get("fact", ""))
    if tool_name == "recall_student_memory":
        return recall_student_memory(session_user_id)
    if tool_name == "create_support_request":
        return create_support_request(args.get("description", ""), session_user_id)
    if tool_name == "request_programme_change":
        return request_programme_change(args.get("requested_program", ""), session_user_id)
    return {"error": f"Unknown tool: {tool_name}"}


def read_token_usage(message) -> Dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    if usage:
        return {
            "prompt_tokens": int(usage.get("input_tokens", 0)),
            "completion_tokens": int(usage.get("output_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
    usage = getattr(message, "response_metadata", {}).get("token_usage", {})
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


@traceable(name="Student Support Agent Run", run_type="chain")
def run_agent(question: str, session_user_id: str,
              secure_mode: bool = False) -> Dict[str, Any]:
    started = time.time()
    system_prompt = (
        "You are a helpful Student Support Assistant. "
        f"The authenticated session identity is {session_user_id}. "
        f"Security mode: {'ENABLED (Assignment 2)' if secure_mode else 'DISABLED (Assignment 1)'}. "
        "Use tools for fee, scholarship, support, memory, and handbook questions. "
        "Never invent policies. If handbook retrieval finds no evidence, say so. "
        "Never request or store passwords, OTPs, PINs, or recovery codes. "
        "Programme changes are sensitive: submit them for staff approval and never claim "
        "they are complete until staff approves them. Be concise and polite."
    )

    # TOOL_SCHEMAS in tools.py are OpenAI-compatible schemas supported by ChatGroq.
    model = build_llm().bind_tools(TOOL_SCHEMAS)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    tools_used: List[Dict[str, Any]] = []
    evidence: List[str] = []
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for _ in range(6):
        response = model.invoke(messages)
        usage = read_token_usage(response)
        for key in tokens:
            tokens[key] += usage[key]
        messages.append(response)

        calls = getattr(response, "tool_calls", []) or []
        if not calls:
            answer = response.content
            if isinstance(answer, list):
                answer = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in answer
                )
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "student_id": session_user_id,
                "secure_mode": secure_mode,
                "question": question,
                "final_answer": str(answer),
                "tools_used": tools_used,
                "retrieved_evidence": evidence,
                "latency_seconds": round(time.time() - started, 2),
                "token_usage": tokens,
            }
            TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TRACE_LOG_PATH, "a", encoding="utf-8") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
            return result

        for call in calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            try:
                tool_result = execute_tool(name, args, session_user_id, secure_mode)
            except Exception as exc:
                tool_result = {"error": f"Tool failed: {type(exc).__name__}"}

            tools_used.append({
                "name": name,
                "arguments": args,
                "result": tool_result,
            })
            if name == "search_student_policy":
                evidence.extend(tool_result.get("evidence", []))

            messages.append(ToolMessage(
                content=json.dumps(tool_result, ensure_ascii=False),
                tool_call_id=call["id"],
            ))

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "student_id": session_user_id,
        "secure_mode": secure_mode,
        "question": question,
        "final_answer": "The agent reached its tool-call limit. Please try a more specific question.",
        "tools_used": tools_used,
        "retrieved_evidence": evidence,
        "latency_seconds": round(time.time() - started, 2),
        "token_usage": tokens,
    }
