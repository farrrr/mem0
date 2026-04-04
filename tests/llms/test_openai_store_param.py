"""Tests for the OpenAI LLM 'store' parameter truthiness check.

Verifies that the 'store' param is only included in API calls when it has
a truthy value, preventing issues with providers (e.g. Cerebras) that don't
support the parameter.
"""

import os
from unittest.mock import Mock, patch, MagicMock

import pytest

from mem0.configs.llms.openai import OpenAIConfig
from mem0.llms.openai import OpenAILLM


@pytest.fixture
def mock_openai_client():
    with patch("mem0.llms.openai.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        yield mock_client


@pytest.fixture(autouse=True)
def clear_openrouter_env():
    """Ensure OPENROUTER_API_KEY is not set during these tests."""
    old = os.environ.pop("OPENROUTER_API_KEY", None)
    yield
    if old is not None:
        os.environ["OPENROUTER_API_KEY"] = old


class TestStoreParamTruthiness:
    """Tests for store param being conditionally included."""

    def test_store_true_included_in_params(self, mock_openai_client):
        """When store=True, it should be passed to the API call."""
        config = OpenAIConfig(
            model="gpt-4.1-nano-2025-04-14", api_key="sk-test", store=True
        )
        llm = OpenAILLM(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "hello"
        mock_response.choices[0].message.tool_calls = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        llm.generate_response(
            messages=[{"role": "user", "content": "hi"}]
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert "store" in call_kwargs
        assert call_kwargs["store"] is True

    def test_store_false_not_included(self, mock_openai_client):
        """When store=False (default), it should NOT be in the API call params."""
        config = OpenAIConfig(
            model="gpt-4.1-nano-2025-04-14", api_key="sk-test", store=False
        )
        llm = OpenAILLM(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "hello"
        mock_response.choices[0].message.tool_calls = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        llm.generate_response(
            messages=[{"role": "user", "content": "hi"}]
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert "store" not in call_kwargs

    def test_store_default_not_included(self, mock_openai_client):
        """Default config should not include store in API call."""
        config = OpenAIConfig(
            model="gpt-4.1-nano-2025-04-14", api_key="sk-test"
        )
        llm = OpenAILLM(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "hello"
        mock_response.choices[0].message.tool_calls = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        llm.generate_response(
            messages=[{"role": "user", "content": "hi"}]
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert "store" not in call_kwargs

    def test_store_none_not_included(self, mock_openai_client):
        """When store is None (falsy), it should NOT be in params."""
        config = OpenAIConfig(
            model="gpt-4.1-nano-2025-04-14", api_key="sk-test"
        )
        config.store = None
        llm = OpenAILLM(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "hello"
        mock_response.choices[0].message.tool_calls = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        llm.generate_response(
            messages=[{"role": "user", "content": "hi"}]
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert "store" not in call_kwargs
