"""LangGraph control-plane nodes, routing, and session orchestration."""
from typing import List
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
try:
    from langsmith import traceable
except Exception:
    def traceable(*args, **kwargs):
        def decorate(fn): return fn
        return decorate

from .config import MAX_PIN_ATTEMPTS, MAX_REVIEW_RETRIES, SUPERVISOR_INTENTS
from .observability import utc_now, get_greeting, redact_pii
from .repositories import get_customer_memory, format_memory_for_prompt, append_customer_memory
from .runtime import LLM, ACCOUNT_STORE
from .guardrails import regex_injection_scan, llm_injection_scan, regex_output_scan, llm_output_scan
from .state import (
    UnifiedAgentState, GuardrailView, IdentityGateView, ContextLoaderView, SupervisorView,
    NetworkAgentView, BillingAgentView, AccountAgentView, EscalationAgentView,
    SupervisorReviewView, OutputGuardrailView, ResponseView, project_into, merge_back,
)
from .agents import (
    NETWORK_AGENT, BILLING_AGENT, ACCOUNT_AGENT, ESCALATION_AGENT, run_specialist,
)

def get_pin_attempts() -> dict:
    """Return this browser session's PIN attempt counter, avoiding cross-user lockouts."""
    if "pin_attempts" not in st.session_state:
        st.session_state["pin_attempts"] = {}
    return st.session_state["pin_attempts"]


@traceable(name="guardrail_node", process_inputs=redact_pii)
def guardrail_node(state: UnifiedAgentState) -> UnifiedAgentState:
    view = project_into(state, GuardrailView)
    query = view.get("query", "")

    regex_flagged, matched = regex_injection_scan(query)
    llm_flagged = llm_injection_scan(query) if not regex_flagged else False
    flagged = regex_flagged or llm_flagged
    layer = "regex" if regex_flagged else ("llm" if llm_flagged else "none")

    log_entry = {
        "timestamp": utc_now(), "node": "GuardrailNode",
        "customer_name": view.get("customer_name", ""),
        "verification_status": view.get("verification_status", "unverified"),
        "query": query[:100], "intent_category": "guardrail",
        "injection_flag": flagged,
        "resolution_type": "blocked" if flagged else "pass",
        "response_summary": (f"Blocked by {layer} layer" if flagged else "Clean"),
    }
    new_log = view.get("decision_log", []) + [log_entry]

    if flagged:
        safe = ("Your request has been flagged for security review. "
                "A human agent will assist you shortly.")
        return merge_back(state, {"injection_flag": True, "agent_response": safe,
                                  "final_response": safe, "decision_log": new_log}, GuardrailView)

    return merge_back(state, {"injection_flag": False, "decision_log": new_log}, GuardrailView)


@traceable(name="identity_gate_node", process_inputs=redact_pii)
def identity_gate_node(state: UnifiedAgentState) -> UnifiedAgentState:
    view = project_into(state, IdentityGateView)
    account_id = view.get("customer_account_id", "")
    supplied_pin = str(view.get("account_pin", ""))
    customer_name = view.get("customer_name", "Unknown")

    account = ACCOUNT_STORE.get(account_id)
    account_status = account["account_status"] if account else "unknown"

    if account is None:
        status, reason = "unverified", "Account not found"
    elif get_pin_attempts().get(account_id, 0) >= MAX_PIN_ATTEMPTS:
        status, reason = "unverified", "Account locked after too many failed attempts"
    elif supplied_pin and supplied_pin == account["account_pin"]:
        status, reason = "verified", "PIN match"
        get_pin_attempts()[account_id] = 0
    else:
        status, reason = "unverified", "PIN missing or incorrect"
        get_pin_attempts()[account_id] = get_pin_attempts().get(account_id, 0) + 1

    log_entry = {
        "timestamp": utc_now(), "node": "IdentityGateNode",
        "customer_name": customer_name, "verification_status": status,
        "query": view.get("query", "")[:100], "intent_category": "identity_check",
        "injection_flag": False,
        "resolution_type": "pass" if status == "verified" else "restrict",
        "response_summary": reason,
    }
    return merge_back(state, {"verification_status": status, "account_status": account_status,
                              "decision_log": view.get("decision_log", []) + [log_entry]},
                      IdentityGateView)


@traceable(name="context_loader_node", process_inputs=redact_pii)
def context_loader_node(state: UnifiedAgentState) -> UnifiedAgentState:
    view = project_into(state, ContextLoaderView)
    if view.get("verification_status") == "verified":
        memory = get_customer_memory(view.get("customer_account_id", ""))
        memory_context = format_memory_for_prompt(memory)
    else:
        memory_context = "No history loaded (customer not verified)."
    return merge_back(state, {"memory_context": memory_context}, ContextLoaderView)


@traceable(name="supervisor_agent_node", process_inputs=redact_pii)
def supervisor_agent_node(state: UnifiedAgentState) -> UnifiedAgentState:
    view = project_into(state, SupervisorView)
    query = view.get("query", "")
    history_text = ""
    for turn in (view.get("conversation_history", []) or [])[-4:]:
        history_text += f"{turn['role'].upper()}: {turn['content'][:100]}\n"

    prompt = (
        "You are a telecom support router for Union Mobile.\n"
        "Classify the customer's intent into exactly one of: network, billing, account, escalation.\n"
        "- network: signal, dropped calls, data, outages\n"
        "- billing: bills, charges, payments, pricing, refunds\n"
        "- account: plan changes, SIM, suspension, profile, ownership\n"
        "- escalation: unresolved, repeated complaint, abusive, out of scope\n\n"
        f"Recent turns:\n{history_text}\n"
        f"Current query: {query}\n\n"
        "Respond with only one word."
    )
    raw = LLM.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    intent = raw if raw in SUPERVISOR_INTENTS else "network"

    log_entry = {
        "timestamp": utc_now(), "node": "SupervisorAgent",
        "customer_name": view.get("customer_name", "Unknown"),
        "verification_status": view.get("verification_status", "unverified"),
        "query": query[:100], "intent_category": intent,
        "injection_flag": False, "resolution_type": "routing",
        "response_summary": f"Routed to {intent} agent",
    }
    return merge_back(state, {"intent_category": intent,
                              "decision_log": view.get("decision_log", []) + [log_entry]},
                      SupervisorView)


def network_agent_node(state: UnifiedAgentState) -> UnifiedAgentState:
    return run_specialist(NETWORK_AGENT, "NetworkAgent", NetworkAgentView, state)


def billing_agent_node(state: UnifiedAgentState) -> UnifiedAgentState:
    return run_specialist(BILLING_AGENT, "BillingAgent", BillingAgentView, state)


def account_agent_node(state: UnifiedAgentState) -> UnifiedAgentState:
    return run_specialist(ACCOUNT_AGENT, "AccountAgent", AccountAgentView, state)


def escalation_agent_node(state: UnifiedAgentState) -> UnifiedAgentState:
    updated = run_specialist(ESCALATION_AGENT, "EscalationAgent", EscalationAgentView, state)
    updated["escalated_already"] = True
    updated["resolution_type"] = "escalate"
    return updated


@traceable(name="supervisor_review_node", process_inputs=redact_pii)
def supervisor_review_node(state: UnifiedAgentState) -> UnifiedAgentState:
    view = project_into(state, SupervisorReviewView)
    query = view.get("query", "")
    agent_response = view.get("agent_response", "")
    verification_status = view.get("verification_status", "unverified")
    retry_count = view.get("retry_count", 0)

    prompt = (
        "You are a support supervisor reviewing an agent reply before it reaches the customer.\n"
        "Approve the reply unless it is clearly off-topic for the customer's question, or clearly harmful or unsafe.\n\n"
        "Important: a reply that correctly declines because the customer is not verified, and asks them to verify, "
        "is a CORRECT and COMPLETE reply. Always APPROVE such a reply. Not answering a request from an unverified "
        "customer is the right outcome, not a failure. Also approve a reply that correctly explains it cannot do "
        "something and offers to escalate to a human.\n"
        "Be lenient: minor imperfections are acceptable.\n\n"
        f"Customer verification status: {verification_status}\n"
        f"Customer question: {query}\n"
        f"Agent reply: {agent_response}\n\n"
        "Answer with exactly one word: APPROVE or REJECT."
    )
    try:
        verdict = LLM.invoke([HumanMessage(content=prompt)]).content.strip().upper()
        approved = verdict.startswith("APPROVE")
    except Exception:
        # Fail closed: after the bounded retry, the workflow routes to human escalation.
        approved = False

    new_retry = retry_count if approved else retry_count + 1

    log_entry = {
        "timestamp": utc_now(), "node": "SupervisorReviewNode",
        "customer_name": view.get("customer_name", "Unknown"),
        "verification_status": verification_status,
        "query": query[:100], "intent_category": view.get("intent_category", ""),
        "injection_flag": False,
        "resolution_type": "approved" if approved else "rejected",
        "response_summary": f"Review {'approved' if approved else 'rejected'} the response",
    }
    return merge_back(state, {"review_approved": approved, "retry_count": new_retry,
                              "decision_log": view.get("decision_log", []) + [log_entry]},
                      SupervisorReviewView)


@traceable(name="output_guardrail_node", process_inputs=redact_pii)
def output_guardrail_node(state: UnifiedAgentState) -> UnifiedAgentState:
    view = project_into(state, OutputGuardrailView)
    response_text = view.get("agent_response", "")
    already_escalated = view.get("escalated_already", False)

    regex_flagged, matched = regex_output_scan(response_text)
    llm_flagged = llm_output_scan(response_text) if not regex_flagged else False
    flagged = regex_flagged or llm_flagged
    layer = "regex" if regex_flagged else ("llm" if llm_flagged else "none")

    log_entry = {
        "timestamp": utc_now(), "node": "OutputGuardrailNode",
        "customer_name": view.get("customer_name", "Unknown"),
        "verification_status": view.get("verification_status", "unverified"),
        "query": view.get("query", "")[:100],
        "intent_category": view.get("intent_category", ""),
        "injection_flag": False, "output_flagged": flagged,
        "resolution_type": "blocked" if flagged else "pass",
        "response_summary": (f"Flagged by {layer} layer" if flagged else "Output clean"),
    }
    new_log = view.get("decision_log", []) + [log_entry]

    if flagged and not already_escalated:
        return merge_back(state, {"output_flagged": True, "decision_log": new_log}, OutputGuardrailView)

    if flagged and already_escalated:
        safe = ("I appreciate your patience. A specialist will follow up with accurate "
                "information for your request shortly.")
        return merge_back(state, {"agent_response": safe, "output_flagged": True,
                                  "decision_log": new_log}, OutputGuardrailView)

    return merge_back(state, {"output_flagged": False, "decision_log": new_log}, OutputGuardrailView)


@traceable(name="response_node", process_inputs=redact_pii)
def response_node(state: UnifiedAgentState) -> UnifiedAgentState:
    view = project_into(state, ResponseView)
    response_text = view.get("agent_response", "I am unable to process your request at this time.")
    intent = view.get("intent_category", "general")
    account_id = view.get("customer_account_id", "")
    query = view.get("query", "")

    updated_history = (view.get("conversation_history", []) or []).copy()
    updated_history.append({"role": "user", "content": query})
    updated_history.append({"role": "assistant", "content": response_text})

    if (account_id and view.get("verification_status") == "verified"
            and not view.get("injection_flag", False)
            and view.get("resolution_type") != "escalate"):
        append_customer_memory(account_id, {
            "timestamp": utc_now(), "query": query[:200], "intent": intent,
            "agent_used": f"{intent.capitalize()} Agent",
            "resolution_type": view.get("resolution_type", "inform"),
            "response_summary": response_text[:200],
        })

    log_entry = {
        "timestamp": utc_now(), "node": "ResponseNode",
        "customer_name": view.get("customer_name", "Unknown"),
        "verification_status": view.get("verification_status", "unverified"),
        "query": query[:100], "intent_category": intent,
        "injection_flag": view.get("injection_flag", False),
        "resolution_type": view.get("resolution_type", "inform"),
        "response_summary": response_text[:100],
    }
    return merge_back(state, {"final_response": response_text,
                              "conversation_history": updated_history,
                              "decision_log": view.get("decision_log", []) + [log_entry]},
                      ResponseView)


# ══════════════════════════════════════════════════════════════════════════════
# 11. ROUTING + GRAPH
# ══════════════════════════════════════════════════════════════════════════════
def route_after_guardrail(state: UnifiedAgentState) -> str:
    return "end" if state.get("injection_flag", False) else "identity_gate"


def route_supervisor_to_agent(state: UnifiedAgentState) -> str:
    return {
        "network": "network_agent",
        "billing": "billing_agent",
        "account": "account_agent",
        "escalation": "escalation_agent",
    }.get(state.get("intent_category", "network"), "network_agent")


def route_after_review(state: UnifiedAgentState) -> str:
    if state.get("review_approved", True):
        return "output_guardrail"
    if state.get("retry_count", 0) <= MAX_REVIEW_RETRIES:
        return "supervisor"
    return "escalation_agent"


def route_after_output_guardrail(state: UnifiedAgentState) -> str:
    if state.get("output_flagged", False) and not state.get("escalated_already", False):
        return "escalation_agent"
    return "response_node"


@st.cache_resource(show_spinner="Compiling support workflow…")
def build_workflow():
    workflow = StateGraph(UnifiedAgentState)

    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("identity_gate", identity_gate_node)
    workflow.add_node("context_loader", context_loader_node)
    workflow.add_node("supervisor", supervisor_agent_node)
    workflow.add_node("network_agent", network_agent_node)
    workflow.add_node("billing_agent", billing_agent_node)
    workflow.add_node("account_agent", account_agent_node)
    workflow.add_node("escalation_agent", escalation_agent_node)
    workflow.add_node("supervisor_review", supervisor_review_node)
    workflow.add_node("output_guardrail", output_guardrail_node)
    workflow.add_node("response_node", response_node)

    workflow.set_entry_point("guardrail")

    workflow.add_conditional_edges("guardrail", route_after_guardrail,
                                   {"identity_gate": "identity_gate", "end": END})
    workflow.add_edge("identity_gate", "context_loader")
    workflow.add_edge("context_loader", "supervisor")
    workflow.add_conditional_edges("supervisor", route_supervisor_to_agent,
                                   {"network_agent": "network_agent", "billing_agent": "billing_agent",
                                    "account_agent": "account_agent", "escalation_agent": "escalation_agent"})
    for agent in ["network_agent", "billing_agent", "account_agent"]:
        workflow.add_edge(agent, "supervisor_review")
    workflow.add_conditional_edges("supervisor_review", route_after_review,
                                   {"output_guardrail": "output_guardrail", "supervisor": "supervisor",
                                    "escalation_agent": "escalation_agent"})
    workflow.add_edge("escalation_agent", "output_guardrail")
    workflow.add_conditional_edges("output_guardrail", route_after_output_guardrail,
                                   {"response_node": "response_node", "escalation_agent": "escalation_agent"})
    workflow.add_edge("response_node", END)

    return workflow.compile()


TELECOM_APP = build_workflow()


def create_initial_state(query: str, customer_name: str = "Guest",
                         customer_account_id: str = "", account_pin: str = "",
                         conversation_history: List[dict] = None) -> UnifiedAgentState:
    return UnifiedAgentState(
        query=query, customer_name=customer_name,
        customer_account_id=customer_account_id, account_pin=account_pin,
        verification_status="unverified", account_status="unknown",
        conversation_history=conversation_history or [],
        memory_context="", intent_category="",
        injection_flag=False, output_flagged=False,
        agent_response="", resolution_type="", tools_used=[], escalation_summary="",
        review_approved=True, retry_count=0, escalated_already=False,
        decision_log=[], final_response="",
    )


def run_query(query: str, **kwargs) -> dict:
    return TELECOM_APP.invoke(create_initial_state(query=query, **kwargs))


# ══════════════════════════════════════════════════════════════════════════════
# 12. SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "messages": [],
        "verified": False,
        "customer_name": "",
        "customer_account_id": "",
        "account_pin": "",
        "conversation_history": [],   # [{role, content}, ...] carried across turns
        "decision_log": [],
        "injection_warned": False,
        "output_warned": False,
        "_fallback_session_id": str(uuid.uuid4()),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


AGENT_DISPLAY = {
    "network": "Network Agent",
    "billing": "Billing Agent",
    "account": "Account Agent",
    "escalation": "Escalation Team",
    "guardrail": "Security System",
}


def process_message(user_input: str) -> dict:
    """Run one turn through the full LangGraph workflow and update session state."""
    customer_name = st.session_state.customer_name or "Guest"
    account_id = st.session_state.customer_account_id if st.session_state.verified else ""
    account_pin = st.session_state.account_pin if st.session_state.verified else ""

    result = run_query(
        query=user_input,
        customer_name=customer_name,
        customer_account_id=account_id,
        account_pin=account_pin,
        conversation_history=st.session_state.conversation_history,
    )

    # Carry conversation history forward for the next turn.
    st.session_state.conversation_history = result.get("conversation_history",
                                                        st.session_state.conversation_history)

    # Append this turn's decision log to the running audit trail.
    st.session_state.decision_log.extend(result.get("decision_log", []))

    intent = result.get("intent_category", "guardrail")
    inj_flag = result.get("injection_flag", False)
    out_flagged = result.get("output_flagged", False)

    if inj_flag:
        st.session_state.injection_warned = True
        agent_name = AGENT_DISPLAY["guardrail"]
    else:
        agent_name = AGENT_DISPLAY.get(intent, "Support Agent")
    if out_flagged:
        st.session_state.output_warned = True

    return {
        "response": result.get("final_response") or result.get("agent_response", "No response."),
        "agent_name": agent_name,
        "intent": intent,
        "resolution_type": result.get("resolution_type", "inform"),
        "injection_flag": inj_flag,
        "output_flagged": out_flagged,
        "tools_used": result.get("tools_used", []),
    }

