"""LLM and embedding client construction."""
from __future__ import annotations
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .config import OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME, EMBEDDING_MODEL

@st.cache_resource(show_spinner="Loading language models…")
def get_models():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured. Set it in environment or Streamlit secrets.")
    common = {"api_key": OPENAI_API_KEY}
    if OPENAI_API_BASE:
        common["base_url"] = OPENAI_API_BASE
    return (
        ChatOpenAI(model=MODEL_NAME, temperature=0, **common),
        OpenAIEmbeddings(model=EMBEDDING_MODEL, **common),
    )
