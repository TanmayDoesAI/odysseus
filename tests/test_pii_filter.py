"""Tests for src/pii_filter — redact/restore round-trip, no model required."""

import pytest
from unittest.mock import patch, MagicMock


def _make_mock_pipeline(entities):
    """Return a mock HF pipeline that returns the given entity list."""
    mock = MagicMock()
    mock.return_value = entities
    return mock


def test_redact_restore_round_trip():
    from src.pii_filter import redact, restore

    entities = [
        {"word": "Peter Parker", "entity_group": "PER", "start": 5, "end": 17, "score": 0.99},
        {"word": "New York", "entity_group": "LOC", "start": 29, "end": 37, "score": 0.98},
        {"word": "+99999", "entity_group": "PHONE", "start": 57, "end": 63, "score": 0.97},
    ]

    with patch("src.pii_filter._pipeline", _make_mock_pipeline(entities)), \
         patch("src.pii_filter._load_error", None):
        text = "I am Peter Parker, living in New York and my number +99999 is broken."
        redacted, mapping = redact(text)

        assert "Peter Parker" not in redacted
        assert "New York" not in redacted
        assert "+99999" not in redacted
        assert len(mapping) == 3

        restored = restore(redacted, mapping)
        assert "Peter Parker" in restored
        assert "New York" in restored
        assert "+99999" in restored


def test_redact_no_entities_returns_unchanged():
    from src.pii_filter import redact

    with patch("src.pii_filter._pipeline", _make_mock_pipeline([])), \
         patch("src.pii_filter._load_error", None):
        text = "Nothing sensitive here."
        redacted, mapping = redact(text)
        assert redacted == text
        assert mapping == {}


def test_restore_empty_mapping_noop():
    from src.pii_filter import restore
    text = "Hello [person 1], how are you?"
    assert restore(text, {}) == text


def test_is_commercial_url():
    from src.pii_filter import is_commercial_url
    assert is_commercial_url("https://api.openai.com/v1/chat/completions")
    assert is_commercial_url("https://api.anthropic.com/v1/messages")
    assert is_commercial_url("https://openrouter.ai/api/v1")
    assert not is_commercial_url("http://localhost:11434")
    assert not is_commercial_url("http://192.168.1.10:11434")


def test_model_load_failure_returns_unchanged():
    from src.pii_filter import redact
    import src.pii_filter as pf

    with patch.object(pf, "_pipeline", None), \
         patch.object(pf, "_load_error", "model not found"):
        text = "My name is John Doe."
        redacted, mapping = redact(text)
        assert redacted == text
        assert mapping == {}
