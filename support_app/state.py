"""Typed workflow state and node-specific projections."""
from dataclasses import dataclass, fields
from typing import List
from typing_extensions import TypedDict

class UnifiedAgentState(TypedDict, total=False):
    customer_name: str
    customer_account_id: str
    account_pin: str
    verification_status: str
    account_status: str

    query: str
    conversation_history: List[dict]
    memory_context: str
    intent_category: str

    injection_flag: bool
    output_flagged: bool

    agent_response: str
    resolution_type: str
    tools_used: List[str]
    escalation_summary: str

    review_approved: bool
    retry_count: int
    escalated_already: bool

    decision_log: List[dict]
    final_response: str


@dataclass
class GuardrailView:
    query: str
    customer_name: str
    verification_status: str
    injection_flag: bool
    agent_response: str
    final_response: str
    decision_log: List[dict]


@dataclass
class IdentityGateView:
    customer_account_id: str
    account_pin: str
    customer_name: str
    query: str
    verification_status: str
    account_status: str
    decision_log: List[dict]


@dataclass
class ContextLoaderView:
    verification_status: str
    customer_account_id: str
    memory_context: str


@dataclass
class SupervisorView:
    query: str
    customer_name: str
    verification_status: str
    conversation_history: List[dict]
    intent_category: str
    decision_log: List[dict]


@dataclass
class NetworkAgentView:
    query: str
    customer_name: str
    intent_category: str
    memory_context: str
    verification_status: str
    customer_account_id: str
    agent_response: str
    resolution_type: str
    tools_used: List[str]
    decision_log: List[dict]


@dataclass
class BillingAgentView:
    query: str
    customer_name: str
    intent_category: str
    memory_context: str
    verification_status: str
    customer_account_id: str
    agent_response: str
    resolution_type: str
    tools_used: List[str]
    decision_log: List[dict]


@dataclass
class AccountAgentView:
    query: str
    customer_name: str
    intent_category: str
    memory_context: str
    verification_status: str
    customer_account_id: str
    agent_response: str
    resolution_type: str
    tools_used: List[str]
    decision_log: List[dict]


@dataclass
class EscalationAgentView:
    query: str
    customer_name: str
    intent_category: str
    memory_context: str
    verification_status: str
    customer_account_id: str
    injection_flag: bool
    agent_response: str
    resolution_type: str
    tools_used: List[str]
    escalation_summary: str
    escalated_already: bool
    decision_log: List[dict]


@dataclass
class SupervisorReviewView:
    query: str
    agent_response: str
    verification_status: str
    customer_name: str
    intent_category: str
    retry_count: int
    review_approved: bool
    decision_log: List[dict]


@dataclass
class OutputGuardrailView:
    agent_response: str
    escalated_already: bool
    customer_name: str
    verification_status: str
    query: str
    intent_category: str
    output_flagged: bool
    decision_log: List[dict]


@dataclass
class ResponseView:
    agent_response: str
    intent_category: str
    customer_account_id: str
    query: str
    verification_status: str
    injection_flag: bool
    resolution_type: str
    conversation_history: List[dict]
    customer_name: str
    final_response: str
    decision_log: List[dict]


def project_into(state: UnifiedAgentState, view_class: type) -> dict:
    view_fields = {f.name for f in fields(view_class)}
    return {k: state.get(k) for k in view_fields if k in state}


def merge_back(state: UnifiedAgentState, agent_output: dict, view_class: type) -> UnifiedAgentState:
    view_fields = {f.name for f in fields(view_class)}
    for k, v in agent_output.items():
        if k in view_fields:
            state[k] = v
    return state


