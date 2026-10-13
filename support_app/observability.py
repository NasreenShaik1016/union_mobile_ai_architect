"""Shared utilities, redaction, and structured logging."""
from __future__ import annotations
from datetime import datetime, timezone
import logging
import re

logger = logging.getLogger("union_mobile")
def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_greeting(customer_name: str) -> str:
    from .config import PLACEHOLDER_NAMES
    if str(customer_name).strip().lower() in PLACEHOLDER_NAMES:
        return "Hello! How can I assist you today?"
    return f"Hello {customer_name}!"

def redact_pii(inputs: dict) -> dict:
    """Redact PINs and common credential fields before trace/log emission."""
    sensitive = {"account_pin", "pin", "password", "api_key", "authorization", "token"}
    def scrub(obj):
        if isinstance(obj, dict):
            return {k: ("[REDACTED]" if k.lower() in sensitive else scrub(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [scrub(x) for x in obj]
        return obj
    return scrub(inputs)
