import json
from pathlib import Path

import streamlit as st

from agent import run_agent
from security import (
    authenticate_user,
    create_demo_accounts,
    load_state,
    resolve_approval_request,
    AUDIT_FILE,
)
from tools import load_students_data

st.set_page_config(page_title="Student Support Agent", page_icon="🎓", layout="wide")
create_demo_accounts()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "auth_role" not in st.session_state:
    st.session_state.auth_role = None
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

st.sidebar.title("Configuration & Identity")
mode = st.sidebar.radio(
    "Assignment Demonstration Mode:",
    ["Assignment 1 (Baseline Agent)", "Assignment 2 (Secure & Production-Aware)"],
)
is_secure_mode = "Assignment 2" in mode
st.sidebar.divider()

if not is_secure_mode:
    st.sidebar.subheader("Assignment 1: Student Context")
    current_user_id = st.sidebar.selectbox(
        "Active Persona Student ID", ["STU1001", "STU1002", "STU1003"]
    )
    user_role = "student"
    st.sidebar.info("Baseline demonstration mode: authorization checks are disabled.")
else:
    st.sidebar.subheader("Assignment 2: Secure Session")
    if not st.session_state.auth_user:
        with st.sidebar.form("login_form"):
            login_id = st.text_input("User ID (STU1001, STU1002, STU1003, or ADMIN)")
            login_pass = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In")
        if submitted:
            result = authenticate_user(login_id, login_pass)
            if result["authenticated"]:
                st.session_state.auth_user = login_id
                st.session_state.auth_role = result["role"]
                st.session_state.chat_history = []
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials.")
        st.info("Sign in from the sidebar to use Assignment 2.")
        st.stop()

    current_user_id = st.session_state.auth_user
    user_role = st.session_state.auth_role
    st.sidebar.success(f"Logged in: {current_user_id} ({user_role})")
    if st.sidebar.button("Log Out"):
        st.session_state.auth_user = None
        st.session_state.auth_role = None
        st.session_state.chat_history = []
        st.session_state.last_trace = None
        st.rerun()

st.title("🎓 Intelligent Student Support Agent")
st.caption(
    f"Mode: {'Assignment 2 (Secure)' if is_secure_mode else 'Assignment 1 (Baseline)'} "
    f"| Actor: {current_user_id}"
)

tab_chat, tab_admin, tab_audit, tab_traces, tab_eval = st.tabs(
    ["💬 Assistant Chat", "🛡️ Staff Approval Panel", "📜 Audit Trail", "🔍 Trace & Metrics", "📊 Final Evaluation & Comparison"]
)

with tab_chat:
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    user_prompt = st.chat_input("Enter your student inquiry...")
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Agent working with tools..."):
                trace = run_agent(
                    question=user_prompt,
                    session_user_id=current_user_id,
                    secure_mode=is_secure_mode,
                )
                st.session_state.last_trace = trace
                st.markdown(trace["final_answer"])
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": trace["final_answer"]}
                )

with tab_admin:
    if is_secure_mode and user_role != "staff":
        st.warning("The staff approval queue is available only to authenticated staff.")
    elif not is_secure_mode:
        st.info("Switch to Assignment 2 and sign in as ADMIN to demonstrate staff approvals.")
    else:
        st.subheader("Human Approval Queue")
        pending = load_state().get("pending_approvals", {})
        pending_items = {
            key: value for key, value in pending.items()
            if value.get("status") == "PENDING"
        }
        if not pending_items:
            st.write("No pending change requests in queue.")
        for request_id, details in pending_items.items():
            with st.expander(
                f"Request: {request_id} - Student: {details['student_id']}",
                expanded=True,
            ):
                st.write(f"**Action:** {details['action_type']}")
                st.write(f"**Requested change:** {details['payload']}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Approve", key=f"approve_{request_id}"):
                        result = resolve_approval_request(
                            request_id, current_user_id, "APPROVE"
                        )
                        st.success(result["message"]) if result["success"] else st.error(result["message"])
                        st.rerun()
                with col2:
                    if st.button("Reject", key=f"reject_{request_id}"):
                        result = resolve_approval_request(
                            request_id, current_user_id, "REJECT"
                        )
                        st.warning(result["message"]) if result["success"] else st.error(result["message"])
                        st.rerun()

        st.divider()
        st.subheader("Student Records (Staff Demo View)")
        st.json(load_students_data())

with tab_audit:
    if is_secure_mode and user_role != "staff":
        st.warning("The audit trail is restricted to authenticated staff.")
    elif not is_secure_mode:
        st.info("Audit review is demonstrated in Assignment 2 using the ADMIN account.")
    else:
        st.subheader("Audit Trail Log")
        if AUDIT_FILE.exists():
            try:
                with open(AUDIT_FILE, "r", encoding="utf-8") as file:
                    audit_records = json.load(file)
                if audit_records:
                    st.dataframe(audit_records, use_container_width=True)
                else:
                    st.write("Audit log is empty.")
            except (json.JSONDecodeError, OSError):
                st.error("Could not read the audit log.")
        else:
            st.write("No audit records.")

with tab_traces:
    st.subheader("Execution Trace & Telemetry")

    # Live execution metrics of the latest run
    if st.session_state.last_trace:
        trace = st.session_state.last_trace
        col1, col2, col3 = st.columns(3)
        col1.metric("Latency", f"{trace['latency_seconds']} s")
        col2.metric("Total Tokens", trace["token_usage"]["total_tokens"])
        col3.metric("Tools Invoked", len(trace["tools_used"]))
        st.markdown("**Tools Executed:**")
        st.json(trace["tools_used"])
        st.markdown("**Retrieved Policy Evidence:**")
        st.json(trace["retrieved_evidence"])

    st.divider()
    st.subheader("All Logged Execution Traces (Structured Table)")
    trace_path = Path("logs/traces.jsonl")
    if trace_path.exists() and trace_path.stat().st_size > 0:
        import pandas as pd

        traces_df = pd.read_json(trace_path, lines=True)
        # Format columns for scannability
        display_df = traces_df[["timestamp", "student_id", "question", "latency_seconds"]]
        display_df["tools_called"] = traces_df["tools_used"].apply(
            lambda tools: ", ".join([t["name"] for t in tools]) if tools else "None"
        )
        display_df["tokens"] = traces_df["token_usage"].apply(lambda tok: tok.get("total_tokens", 0))
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No traces logged to logs/traces.jsonl yet.")


with tab_eval:
    st.subheader("📊 Final Evaluation: Assignment 1 vs. Assignment 2 Comparison")
    st.caption("Measurable criteria comparing baseline functionality against security controls.")

    comparison_file = Path("reports/assignment2_comparison_evaluation.csv")
    if comparison_file.exists():
        import pandas as pd
        comp_df = pd.read_csv(comparison_file)
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Run 'python evaluation.py' in the terminal to generate the comparison report.")

    st.divider()
    st.subheader("Automated Unit Test Results (10/10 Scenarios)")
    a1_file = Path("reports/assignment1_evaluation.csv")
    if a1_file.exists():
        import pandas as pd
        a1_df = pd.read_csv(a1_file)
        st.dataframe(a1_df, use_container_width=True)
