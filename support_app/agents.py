"""Specialist agent prompts and construction."""
from typing import List
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.managed import RemainingSteps
from .config import GROUNDING_RULES, HIGH_RISK_OPERATIONS
from .runtime import LLM
from .tools import search_policy, get_plan, raise_network_ticket, escalate_to_human
from .state import UnifiedAgentState, project_into, merge_back
from .observability import get_greeting, utc_now

class AgentState(MessagesState):
    verification_status: str
    customer_account_id: str
    injection_flag: bool
    customer_name: str
    query: str
    remaining_steps: RemainingSteps


NETWORK_PROMPT = (
    "You are a Senior Network Support Specialist at Union Mobile. "
    "You help customers resolve connectivity problems such as dropped calls, weak signal, slow data, and outages.\n\n"
    "HOW TO WORK:\n"
    "1. Start by understanding the specific symptom. If it is unclear, ask a brief clarifying question.\n"
    "2. Use search_policy to align your troubleshooting with Union Mobile's network support guidance, and offer "
    "clear, practical, step-by-step advice. General troubleshooting does not require verification, so you can help "
    "any customer with it.\n"
    "3. Use raise_network_ticket only when the issue cannot reasonably be resolved remotely. Raising a ticket is an "
    "account-specific action, so if the customer is not verified the tool will deny it; in that case, provide the "
    "general troubleshooting steps and explain that raising a ticket requires verification.\n"
    "4. If troubleshooting clearly will not help and the customer needs a person, use escalate_to_human.\n\n"
    "Be concrete and reassuring. Give steps the customer can actually follow, and set a realistic expectation for "
    "any follow-up. " + GROUNDING_RULES
)

BILLING_PROMPT = (
    "You are a Senior Billing Specialist at Union Mobile with deep knowledge of plans, charges, and refund policy. "
    "Your job is to help verified customers understand and resolve billing questions accurately and calmly.\n\n"
    "HOW TO WORK:\n"
    "1. For any question about charges, usage, allowances, or pricing, first call get_plan to base your answer "
    "on the customer's real plan record. Never guess at numbers.\n"
    "2. For any refund or credit request, first call search_policy to check whether Union Mobile policy allows a "
    "refund for that situation. Read the returned policy carefully.\n"
    "   - If policy supports the refund, tell the customer their request is valid, then use escalate_to_human so "
    "the billing team can process it. Explain that an agent cannot issue refunds directly.\n"
    "   - If policy does not support the refund, politely decline and explain the specific reason, based on the "
    "policy you retrieved. Offer any appropriate alternative.\n"
    "3. If the customer is not verified, a data tool will deny access. In that case, explain that billing details "
    "require verification and do not attempt to work around it.\n\n"
    "NEVER invent charges, credits, refund outcomes, or policy. NEVER promise a refund amount or timeline yourself; "
    "the billing team owns that once escalated. " + GROUNDING_RULES
)

ACCOUNT_PROMPT = (
    "You are a Senior Account Management Specialist at Union Mobile. "
    "You help verified customers understand and manage their account: plan details, upgrade and downgrade options, "
    "profile and contact preferences, and general account questions.\n\n"
    "HOW TO WORK:\n"
    "1. Use get_plan to ground any answer about the customer's current plan, cost, or contract. Never guess.\n"
    "2. Use search_policy when the customer asks what is allowed or how a process works, and base your explanation "
    "on the retrieved policy.\n"
    "3. HIGH-RISK OPERATIONS: the following operations are high-risk and you must NOT attempt them yourself: "
    f"{', '.join(HIGH_RISK_OPERATIONS)}. These require senior manager authorization. "
    "Use escalate_to_human with a clear reason and an appropriate urgency, and tell the customer a senior manager "
    "will handle it. This holds even for a fully verified customer.\n"
    "4. If the customer is not verified, a data tool will deny access. Explain that account details require "
    "verification and do not try to work around it.\n\n"
    "You can explain and advise on routine account matters. You never execute account changes yourself; anything "
    "that modifies the account is either explained and left to the customer or escalated to a human. " + GROUNDING_RULES
)

ESCALATION_PROMPT = (
    "You are the Escalation Specialist at Union Mobile. You handle cases that a specialist agent cannot resolve and "
    "that need a human: unresolved issues, requests beyond agent authority, high-risk account operations, and "
    "situations flagged for review.\n\n"
    "HOW TO WORK:\n"
    "1. Use escalate_to_human exactly once, with a clear, specific reason and an appropriate urgency level "
    "(Low, Medium, or High).\n"
    "2. After escalating, give the customer a brief, warm message that confirms their case has been handed to a "
    "human specialist and that they will be followed up.\n\n"
    "Do not attempt to resolve the underlying issue yourself, and do not make promises about the outcome. Your role "
    "is a clean, reassuring handoff. " + GROUNDING_RULES
)


@st.cache_resource(show_spinner="Preparing specialist agents…")
def build_specialists():
    network = create_react_agent(
        LLM, [search_policy, raise_network_ticket, escalate_to_human],
        state_schema=AgentState, prompt=NETWORK_PROMPT,
    )
    billing = create_react_agent(
        LLM, [search_policy, get_plan, escalate_to_human],
        state_schema=AgentState, prompt=BILLING_PROMPT,
    )
    account = create_react_agent(
        LLM, [search_policy, get_plan, escalate_to_human],
        state_schema=AgentState, prompt=ACCOUNT_PROMPT,
    )
    escalation = create_react_agent(
        LLM, [search_policy, escalate_to_human],
        state_schema=AgentState, prompt=ESCALATION_PROMPT,
    )
    return network, billing, account, escalation


NETWORK_AGENT, BILLING_AGENT, ACCOUNT_AGENT, ESCALATION_AGENT = build_specialists()


def derive_resolution(messages: List, intent: str) -> str:
    tool_names, tool_outputs = [], []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            tool_names += [tc["name"] for tc in m.tool_calls]
        if isinstance(m, ToolMessage):
            tool_outputs.append(str(m.content))
    if any("ACCESS DENIED" in o for o in tool_outputs):
        return "blocked"
    if "escalate_to_human" in tool_names:
        return "escalate"
    if "raise_network_ticket" in tool_names:
        return "troubleshoot"
    if intent == "network":
        return "troubleshoot"
    return "inform"


def collect_tools_used(messages: List) -> List[str]:
    used = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            used += [tc["name"] for tc in m.tool_calls]
    return used


def run_specialist(agent, agent_label: str, view_class: type, state: UnifiedAgentState) -> UnifiedAgentState:
    view = project_into(state, view_class)
    greeting = get_greeting(view.get("customer_name", "Guest"))
    user_message = (
        f"{greeting}\n\n"
        f"Customer query: {view.get('query', '')}\n\n"
        f"{view.get('memory_context', '')}"
    )
    result = agent.invoke({
        "messages": [HumanMessage(content=user_message)],
        "verification_status": view.get("verification_status", "unverified"),
        "customer_account_id": view.get("customer_account_id", ""),
        "injection_flag": view.get("injection_flag", False),
        "customer_name": view.get("customer_name", "Guest"),
        "query": view.get("query", ""),
    })
    messages = result["messages"]
    response_text = messages[-1].content
    tools_used = collect_tools_used(messages)
    resolution = derive_resolution(messages, view.get("intent_category", ""))

    log_entry = {
        "timestamp": utc_now(), "node": agent_label,
        "customer_name": view.get("customer_name", "Unknown"),
        "verification_status": view.get("verification_status", "unverified"),
        "query": view.get("query", "")[:100],
        "intent_category": view.get("intent_category", ""),
        "injection_flag": view.get("injection_flag", False),
        "resolution_type": resolution,
        "tools_used": tools_used,
        "response_summary": response_text[:100],
    }
    return merge_back(
        state,
        {"agent_response": response_text, "resolution_type": resolution,
         "tools_used": tools_used,
         "decision_log": view.get("decision_log", []) + [log_entry]},
        view_class,
    )

