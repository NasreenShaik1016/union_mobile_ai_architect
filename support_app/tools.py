"""LangGraph tools with verification checks at the tool boundary."""
from typing import Annotated
import uuid
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from .runtime import ACCOUNT_STORE, PLAN_STORE, search_policy as search_policy_kb
from .config import TICKETS_FILE, ESCALATIONS_FILE
from .repositories import append_record
from .repositories import append_customer_memory
from .observability import utc_now

@tool
def search_policy(query: str) -> str:
    """Search Union Mobile support policy for guidance relevant to the query. Returns the most relevant policy text."""
    return search_policy_kb(query)


@tool
def get_plan(state: Annotated[dict, InjectedState]) -> str:
    """Get the plan details (plan name, cost, data allowance, data used, contract) for the current verified customer."""
    if state.get("verification_status") != "verified":
        return ("ACCESS DENIED: the customer is not verified. Ask them to verify with their "
                "account PIN before sharing plan details.")
    account_id = state.get("customer_account_id", "")
    plan = PLAN_STORE.get(account_id)
    if plan is None:
        return "No plan is on record for this account."
    return (
        f"Plan: {plan['plan_name']} | Monthly cost: ${plan['monthly_cost_usd']} | "
        f"Data allowance: {plan['data_allowance_gb']} GB | Data used this cycle: {plan['data_used_gb']} GB | "
        f"Voice minutes: {plan['voice_minutes']} | Contract ends: {plan['contract_end_date']} | "
        f"Roaming enabled: {plan['roaming_enabled']}"
    )


@tool
def raise_network_ticket(issue: str, state: Annotated[dict, InjectedState]) -> str:
    """Raise a network field-technician ticket for the verified customer's line."""
    if state.get("verification_status") != "verified":
        return ("ACCESS DENIED: raising a ticket requires verification. Offer general "
                "troubleshooting steps instead.")
    ticket_id = f"NET-{uuid.uuid4().hex[:8].upper()}"
    try:
        append_record(TICKETS_FILE, {
            "ticket_id": ticket_id, "created_at": utc_now(),
            "customer_account_id": state.get("customer_account_id", ""),
            "issue": issue[:500], "status": "pending_human_follow_up",
        })
    except Exception:
        return "TICKET NOT CREATED: the ticket queue is unavailable. Please escalate to a human agent."
    return f"TICKET RECORDED: {ticket_id}. It is queued for field-technician follow-up."


@tool
def escalate_to_human(reason: str, urgency: str, state: Annotated[dict, InjectedState]) -> str:
    """Escalate the case to a human agent with a reason and an urgency level (Low, Medium, or High)."""
    is_security_incident = bool(state.get("injection_flag", False))
    final_urgency = "High" if is_security_incident else urgency

    packet = {
        "timestamp": utc_now(),
        "customer_name": state.get("customer_name", "Unknown"),
        "customer_account_id": state.get("customer_account_id", "N/A"),
        "verification_status": state.get("verification_status", "unverified"),
        "reason": reason,
        "urgency": final_urgency,
        "security_incident": is_security_incident,
        "query": state.get("query", "")[:200],
    }

    try:
        append_record(ESCALATIONS_FILE, packet)
    except Exception:
        return "ESCALATION QUEUE UNAVAILABLE: please contact a human support representative directly."

    if state.get("verification_status") == "verified" and not is_security_incident:
        account_id = state.get("customer_account_id", "")
        if account_id:
            append_customer_memory(account_id, {
                "timestamp": packet["timestamp"],
                "query": packet["query"],
                "intent": "escalation",
                "agent_used": "Escalation Team",
                "resolution_type": "escalate",
                "response_summary": f"Escalated to human. Reason: {reason}. Urgency: {final_urgency}.",
            })

    tag = " [SECURITY INCIDENT]" if is_security_incident else ""
    return f"ESCALATED{tag}: case handed to a human agent at {final_urgency} urgency. Reason recorded: {reason}."


