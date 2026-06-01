"""PII redaction layer using openai/privacy-filter.

Intercepts outgoing messages to commercial LLMs, replaces detected PII with
labeled placeholders, and restores originals in the response. Opt-in via the
`pii_filter` feature flag (disabled by default).

Model: https://huggingface.co/openai/privacy-filter
Defaults to the q4 quantized ONNX variant (~917 MB) for low-compute hardware.
Falls back to the full model if optimum is not installed.
"""

import contextvars
import logging
import re
from typing import Optional

# Carries the active redaction mapping through async task context so
# chat_routes.py can restore the full streamed response without needing
# stream_llm to return a value.
active_pii_mapping: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "active_pii_mapping", default={}
)

logger = logging.getLogger(__name__)

_pipeline = None
_load_error: Optional[str] = None

_LABEL_DISPLAY = {
    "PER": "person",
    "LOC": "location",
    "ORG": "organization",
    "PHONE": "phone number",
    "EMAIL": "email address",
    "DATE": "date",
    "ID": "ID",
    "MISC": "info",
}


def _load_pipeline():
    global _pipeline, _load_error
    if _pipeline is not None or _load_error is not None:
        return

    try:
        # Prefer quantized ONNX (model_q4, ~917 MB) via optimum for low-compute hardware.
        try:
            from optimum.onnxruntime import ORTModelForTokenClassification
            from transformers import AutoTokenizer, pipeline as hf_pipeline

            model_id = "openai/privacy-filter"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = ORTModelForTokenClassification.from_pretrained(
                model_id, file_name="onnx/model_q4.onnx"
            )
            _pipeline = hf_pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
            logger.info("pii_filter: loaded quantized ONNX model (model_q4)")
        except Exception:
            # optimum not available — fall back to full transformers model
            from transformers import pipeline as hf_pipeline

            _pipeline = hf_pipeline(
                "token-classification",
                model="openai/privacy-filter",
                aggregation_strategy="simple",
            )
            logger.info("pii_filter: loaded full transformers model")

    except Exception as e:
        _load_error = str(e)
        logger.warning(f"pii_filter: failed to load model, filter disabled: {e}")


def _label_to_display(entity_group: str, index: int) -> str:
    base = _LABEL_DISPLAY.get(entity_group.upper(), entity_group.lower())
    return f"[{base} {index}]"


def redact(text: str) -> tuple[str, dict]:
    """Replace PII spans with placeholders. Returns (redacted_text, mapping).

    mapping maps placeholder -> original value so restore() can put them back.
    Returns (text, {}) unchanged if the model isn't loaded or encounters an error.
    """
    _load_pipeline()
    if _pipeline is None:
        return text, {}

    try:
        entities = _pipeline(text)
    except Exception as e:
        logger.warning(f"pii_filter: inference failed, passing text unchanged: {e}")
        return text, {}

    if not entities:
        return text, {}

    # Build placeholder map, deduplicating identical spans.
    seen: dict[str, str] = {}  # original -> placeholder
    counters: dict[str, int] = {}

    for ent in sorted(entities, key=lambda e: e["start"]):
        word = ent["word"].strip()
        if not word or word in seen:
            continue
        group = ent.get("entity_group", "MISC")
        counters[group] = counters.get(group, 0) + 1
        placeholder = _label_to_display(group, counters[group])
        seen[word] = placeholder

    if not seen:
        return text, {}

    mapping = {v: k for k, v in seen.items()}  # placeholder -> original
    redacted = text
    # Replace longest matches first to avoid partial-overlap issues.
    for original in sorted(seen, key=len, reverse=True):
        redacted = redacted.replace(original, seen[original])

    return redacted, mapping


def restore(text: str, mapping: dict) -> str:
    """Replace placeholders back with originals."""
    if not mapping:
        return text
    result = text
    for placeholder, original in mapping.items():
        result = result.replace(placeholder, original)
    return result


def is_commercial_url(url: str) -> bool:
    """Return True if the URL points to a known commercial provider."""
    u = url.lower()
    return any(
        host in u
        for host in (
            "api.openai.com",
            "api.anthropic.com",
            "openrouter.ai",
            "groq.com",
            "together.ai",
            "fireworks.ai",
            "cohere.com",
            "mistral.ai",
        )
    )
