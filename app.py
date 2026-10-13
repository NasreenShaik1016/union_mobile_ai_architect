"""Streamlit entry point for Union Mobile AI Support."""
from datetime import datetime
import logging
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Union Mobile AI Support",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

from support_app.config import (
    DATA_DIR, MAX_PIN_ATTEMPTS, POLICY_FILE, MEMORY_FILE, LANGCHAIN_API_KEY,
    LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT,
)
from support_app.observability import configure_logging, utc_now, logger
from support_app.repositories import get_customer_memory, load_memory_store, save_memory_store
from support_app.runtime import ACCOUNT_STORE, PLAN_STORE, ACCOUNTS_DF, POLICY_COUNT
from support_app.workflow import (
    get_pin_attempts, init_session, process_message,
)

configure_logging()
# ══════════════════════════════════════════════════════════════════════════════
# 13. UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="main-header">
    <h1 style="margin:0;font-size:1.8em;">📱 Union Mobile AI Customer Support</h1>
    <p style="margin:6px 0 0 0;opacity:.85;">Secure multi-agent support · LangGraph · Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── SIDEBAR: login / verification ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 Customer Login")

    sel_id = st.text_input("Account ID", placeholder="Enter your account ID").strip()
    inp_name = st.text_input("Customer Name", placeholder="Enter your full name")
    inp_pin = st.text_input("Account PIN", type="password", placeholder="4-digit PIN")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Verify", type="primary", use_container_width=True):
            if not sel_id:
                st.error("Select an Account ID")
            elif not inp_name or not inp_pin:
                st.error("Enter name and PIN")
            else:
                account = ACCOUNT_STORE.get(sel_id)
                if account is None:
                    st.error("Account not found")
                elif get_pin_attempts().get(sel_id, 0) >= MAX_PIN_ATTEMPTS:
                    st.error("🔒 Account locked after too many failed attempts.")
                else:
                    name_ok = inp_name.strip().lower() == str(account["customer_name"]).strip().lower()
                    pin_ok = inp_pin.strip() == account["account_pin"]
                    if name_ok and pin_ok:
                        get_pin_attempts()[sel_id] = 0
                        st.session_state.verified = True
                        st.session_state.customer_name = account["customer_name"]
                        st.session_state.customer_account_id = sel_id
                        st.session_state.account_pin = inp_pin.strip()
                        st.success("✅ Verified!")
                        st.rerun()
                    else:
                        get_pin_attempts()[sel_id] = get_pin_attempts().get(sel_id, 0) + 1
                        remaining = MAX_PIN_ATTEMPTS - get_pin_attempts()[sel_id]
                        if remaining > 0:
                            st.error(f"Name or PIN incorrect ({remaining} attempt(s) left)")
                        else:
                            st.error("🔒 Account now locked for this session.")
    with c2:
        if st.button("🔓 Guest", use_container_width=True):
            st.session_state.verified = False
            st.session_state.customer_name = inp_name or "Guest"
            st.session_state.customer_account_id = ""
            st.session_state.account_pin = ""
            st.info("Guest mode")
            st.rerun()

    st.markdown("### Status")
    if st.session_state.verified:
        st.markdown(f'<span class="badge-verified">✅ VERIFIED — {st.session_state.customer_name}</span>',
                    unsafe_allow_html=True)
        st.caption(f"Account ID: {st.session_state.customer_account_id}")
    elif get_pin_attempts().get(sel_id, 0) >= MAX_PIN_ATTEMPTS and sel_id:
        st.markdown('<span class="badge-locked">🔒 LOCKED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-unverified">❌ NOT VERIFIED</span>', unsafe_allow_html=True)

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        if st.button("🔄 Reset Session", use_container_width=True):
            for k in ["messages", "verified", "customer_name", "customer_account_id",
                      "account_pin", "conversation_history", "decision_log",
                      "injection_warned", "output_warned"]:
                st.session_state.pop(k, None)
            init_session()
            st.rerun()
    with c4:
        if st.button("🗑️ Clear History", use_container_width=True):
            acct = st.session_state.customer_account_id
            if acct:
                store = load_memory_store()
                store.pop(acct, None)
                save_memory_store(store)
            st.session_state.conversation_history = []
            st.success("Cleared")

    st.markdown("### 💡 Test Credentials")
    if not ACCOUNTS_DF.empty:
        s = ACCOUNTS_DF.iloc[0]
        st.code(f"ID:   {s['customer_account_id']}\nName: {s['customer_name']}\nPIN:  {s['account_pin']}")

    st.caption(f"📚 {POLICY_COUNT} policy sections loaded.")


# ── MAIN: chat + info column ──────────────────────────────────────────────────
col_chat, col_info = st.columns([2, 1])

with col_chat:
    if st.session_state.injection_warned:
        st.markdown('<div class="injection-warning">⚠️ <b>Input Security Alert:</b> A prompt injection attempt was detected and blocked.</div>',
                    unsafe_allow_html=True)
    if st.session_state.output_warned:
        st.markdown('<div class="output-warning">🔴 <b>Output Safety Alert:</b> A generated response was intercepted for policy compliance.</div>',
                    unsafe_allow_html=True)

    st.markdown("### 💬 Chat")

    if not st.session_state.messages:
        if st.session_state.verified:
            welcome = (f"Hello {st.session_state.customer_name}! I'm your Union Mobile AI support "
                       "assistant. I can help with network issues, billing questions, and account management.")
        else:
            welcome = ("Welcome to Union Mobile support! I can help with general network troubleshooting. "
                       "For billing and account access, please verify your identity in the login panel.")
        st.session_state.messages.append(
            {"role": "assistant", "content": welcome, "agent": "🤖 Support Assistant", "timestamp": utc_now()}
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("agent"):
                st.caption(msg["agent"])
            st.write(msg["content"])

    if user_input := st.chat_input("Type your message here…"):
        st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": utc_now()})
        with st.spinner("Routing to the right specialist…"):
            try:
                result = process_message(user_input)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["response"],
                     "agent": result["agent_name"], "timestamp": utc_now()}
                )
            except Exception:
                logger.exception("Unhandled error while processing customer message")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "A technical issue occurred. Please try again later.",
                     "agent": "⚠️ System", "timestamp": utc_now()}
                )
        st.rerun()


with col_info:
    st.markdown("### 📋 Interaction History")
    with st.expander("View Past Interactions", expanded=False):
        acct = st.session_state.customer_account_id
        if acct:
            memory = get_customer_memory(acct)
            if memory:
                for m in reversed(memory):
                    ts = m.get("timestamp", "")[:10]
                    st.markdown(
                        f"**{ts}** — _{m.get('intent', '').upper()}_\n"
                        f"- 🤖 {m.get('agent_used', '')}\n"
                        f"- ✓ {m.get('resolution_type', '')}\n"
                        f"- 💬 _{m.get('query', '')[:60]}…_\n---"
                    )
            else:
                st.info("No past interactions found.")
        else:
            st.info("Verify your identity to view history.")

    if st.session_state.verified and st.session_state.customer_account_id:
        st.markdown("### 👤 Account")
        acct = st.session_state.customer_account_id
        account = ACCOUNT_STORE.get(acct, {})
        plan = PLAN_STORE.get(acct, {})
        with st.expander("Details", expanded=True):
            st.write(f"**Name:** {account.get('customer_name', 'N/A')}")
            st.write(f"**Account ID:** {acct}")
            st.write(f"**Status:** {account.get('account_status', 'N/A')}")
            if plan:
                st.write(f"**Plan:** {plan.get('plan_name', 'N/A')}")
                st.write(f"**Monthly:** ${plan.get('monthly_cost_usd', 'N/A')}")

    st.markdown("### ⚡ Quick Actions")
    actions = {
        "📶 Signal issue": "My signal keeps dropping and I keep losing calls. What can I do?",
        "💳 Check bill": "Can you explain why my bill is higher than usual this month?",
        "📱 Change plan": "I'd like to upgrade to a plan with more data.",
        "🆘 Get help": "I've had this issue unresolved for three weeks now.",
    }
    for label, msg in actions.items():
        if st.button(label, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": msg, "timestamp": utc_now()})
            with st.spinner("Processing…"):
                try:
                    result = process_message(msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["response"],
                         "agent": result["agent_name"], "timestamp": utc_now()}
                    )
                except Exception:
                    logger.exception("Unhandled error while processing quick action")
                    st.error("A technical issue occurred. Please try again later.")
            st.rerun()

    turns = len(st.session_state.conversation_history) // 2
    if turns > 0:
        st.markdown("### 🔄 Conversation")
        st.metric("Turns in session", turns)
        st.caption("Full history is carried forward into each turn.")


# ── DECISION LOG (audit trail) ────────────────────────────────────────────────
st.divider()
st.markdown("### 📊 Decision Log (Audit Trail)")
with st.expander("View Full Decision Log", expanded=False):
    if st.session_state.decision_log:
        COLORS = {
            "network": "#E3F2FD", "billing": "#E8F5E9", "account": "#FFF3E0",
            "escalation": "#FFEBEE", "guardrail": "#F3E5F5",
            "identity_check": "#E0F2F1", "routing": "#F5F5F5",
        }
        df_log = pd.DataFrame(st.session_state.decision_log)

        ca, cb, cc, cd = st.columns(4)
        ca.metric("Total Nodes", len(df_log))
        cb.metric("Injections Blocked",
                  int(df_log.get("injection_flag", pd.Series([False] * len(df_log))).sum()))
        cc.metric("Output Intercepts",
                  int(df_log.get("output_flagged", pd.Series([False] * len(df_log))).sum()))
        cd.metric("Escalations",
                  int((df_log.get("resolution_type", pd.Series([""] * len(df_log))) == "escalate").sum()))

        dcols = ["timestamp", "node", "customer_name", "verification_status",
                 "intent_category", "injection_flag", "resolution_type", "response_summary"]
        show = [c for c in dcols if c in df_log.columns]

        def color_row(row):
            bg = COLORS.get(row.get("intent_category", ""), "#FFFFFF")
            return [f"background-color:{bg}"] * len(row)

        st.dataframe(df_log[show].style.apply(color_row, axis=1),
                     use_container_width=True, height=320)

        csv = df_log[show].to_csv(index=False)
        st.download_button("⬇️ Download Audit Log (CSV)", data=csv,
                           file_name=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
        st.markdown("**Colors:** 🔵 Network · 🟢 Billing · 🟠 Account · 🔴 Escalation · 🟣 Guardrail · 🟩 Identity")
    else:
        st.info("No log entries yet. Start a conversation.")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
<div style="text-align:center;color:#888;font-size:.8em;padding:10px">
    Union Mobile AI Support · LangGraph 11-node workflow · Chroma policy search · Streamlit Secrets
    <br>⚠️ Demonstration system only.
</div>
""",
    unsafe_allow_html=True,
)
