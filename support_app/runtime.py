"""Application resource composition root."""
from .models import get_models
from .repositories import load_stores, build_policy_vectorstore, search_policy_kb

LLM, EMBEDDINGS = get_models()
ACCOUNT_STORE, PLAN_STORE, ACCOUNTS_DF = load_stores()
POLICY_VECTORSTORE, POLICY_COUNT = build_policy_vectorstore()

def search_policy(query: str, top_k: int = 3) -> str:
    return search_policy_kb(POLICY_VECTORSTORE, query, top_k)
