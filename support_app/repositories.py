"""Data access for demo CSVs, policy retrieval, and customer memory.

For production, replace the CSV/JSON adapters with managed database and object-storage
implementations. The JSON adapter uses atomic writes to avoid partial files.
"""
from __future__ import annotations
import json
import os
import re
import threading
from pathlib import Path
from typing import List
import pandas as pd
import streamlit as st
from langchain_core.documents import Document
from .config import ACCOUNTS_FILE, PLANS_FILE, POLICY_FILE, MEMORY_FILE, POLICY_TOP_K, POLICY_SIMILARITY_FLOOR
from .models import get_models

_memory_lock = threading.RLock()

def load_memory_store() -> dict:
    if not MEMORY_FILE.exists():
        return {}
    try:
        with _memory_lock, MEMORY_FILE.open("r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, OSError):
        # Fail closed: do not overwrite a corrupt store without operator awareness.
        raise RuntimeError(f"Customer memory store is unreadable: {MEMORY_FILE}")

def save_memory_store(store: dict) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = MEMORY_FILE.with_suffix(MEMORY_FILE.suffix + ".tmp")
    with _memory_lock:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, MEMORY_FILE)

def get_customer_memory(customer_account_id: str, limit: int = 5) -> List[dict]:
    if not customer_account_id:
        return []
    store = load_memory_store()
    return store.get(customer_account_id, [])[-limit:]

def append_customer_memory(customer_account_id: str, interaction: dict) -> None:
    if not customer_account_id:
        return
    store = load_memory_store()
    store.setdefault(customer_account_id, []).append(interaction)
    # Bound growth; long-term memory should be governed by retention policy.
    store[customer_account_id] = store[customer_account_id][-100:]
    save_memory_store(store)

def format_memory_for_prompt(memory: List[dict]) -> str:
    if not memory:
        return "No previous interactions on record."
    lines = ["Previous interactions:"]
    for item in memory:
        lines.append(
            f"[{item.get('timestamp', '')[:10]}] {item.get('intent', '')} / "
            f"{item.get('resolution_type', '')}: {item.get('response_summary', '')[:160]}"
        )
    return "\n".join(lines)

@st.cache_data(show_spinner="Loading account & plan data…")
def load_stores():
    for path in (ACCOUNTS_FILE, PLANS_FILE, POLICY_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")
    accounts_df = pd.read_csv(ACCOUNTS_FILE, dtype={"account_pin": str})
    required_accounts = {"customer_account_id", "customer_name", "account_pin", "account_status",
                         "autopay_enabled", "date_joined"}
    if not required_accounts.issubset(accounts_df.columns):
        raise ValueError(f"accounts.csv missing columns: {sorted(required_accounts - set(accounts_df.columns))}")
    if accounts_df["customer_account_id"].duplicated().any():
        raise ValueError("accounts.csv contains duplicate customer_account_id values")
    account_store = {
        r["customer_account_id"]: {
            "customer_name": r["customer_name"], "account_pin": str(r["account_pin"]),
            "account_status": r["account_status"], "autopay_enabled": r["autopay_enabled"],
            "date_joined": r["date_joined"],
        } for _, r in accounts_df.iterrows()
    }
    plans_df = pd.read_csv(PLANS_FILE)
    required_plans = {"customer_account_id", "plan_name", "monthly_cost_usd", "data_allowance_gb",
                      "data_used_gb", "voice_minutes", "contract_end_date", "roaming_enabled"}
    if not required_plans.issubset(plans_df.columns):
        raise ValueError(f"plans.csv missing columns: {sorted(required_plans - set(plans_df.columns))}")
    plan_store = {r["customer_account_id"]: {
        "plan_name": r["plan_name"], "monthly_cost_usd": r["monthly_cost_usd"],
        "data_allowance_gb": r["data_allowance_gb"], "data_used_gb": r["data_used_gb"],
        "voice_minutes": r["voice_minutes"], "contract_end_date": r["contract_end_date"],
        "roaming_enabled": r["roaming_enabled"],
    } for _, r in plans_df.iterrows()}
    return account_store, plan_store, accounts_df

def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader
    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)

@st.cache_resource(show_spinner="Building policy knowledge base…")
def build_policy_vectorstore():
    from langchain_chroma import Chroma
    _, embeddings = get_models()
    text = _extract_pdf_text(POLICY_FILE)
    if "## POLICY:" in text:
        chunks = ["POLICY:" + s.strip() for s in text.split("## POLICY:")[1:] if s.strip()]
    else:
        chunks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not chunks:
        raise ValueError("No text could be extracted from policy_kb.pdf")
    documents = [Document(page_content=chunk, metadata={"source": POLICY_FILE.name, "section": i})
                 for i, chunk in enumerate(chunks, 1)]
    store = Chroma.from_documents(documents, embedding=embeddings, collection_name="union_mobile_policy")
    return store, len(chunks)

def search_policy_kb(vectorstore, query: str, top_k: int = POLICY_TOP_K) -> str:
    if not query or not query.strip():
        return "No specific policy section was found for this query."
    results = vectorstore.similarity_search_with_relevance_scores(query=query, k=top_k)
    selected = [doc.page_content for doc, score in results if score >= POLICY_SIMILARITY_FLOOR]
    if not selected:
        return "No specific policy section was found for this query."
    return "\n\n".join(selected)


def append_record(path: Path, record: dict) -> None:
    """Append a record to a local demo queue using atomic replacement.

    This adapter is safe against partial file writes, but is not a distributed queue.
    Replace with Azure SQL/Cosmos DB plus Service Bus for multi-instance production.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _memory_lock
    with lock:
        records = []
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    records = loaded
            except (json.JSONDecodeError, OSError) as exc:
                raise RuntimeError(f"Queue file is unreadable: {path}") from exc
        records.append(record)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
