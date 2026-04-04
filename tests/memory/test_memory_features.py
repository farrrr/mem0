"""Tests for Memory features: system message filtering, fallback LLM retry.

Covers:
- fix(core): filter system messages from extraction
- feat(core): add fallback LLM with 3-attempt retry
"""

import json
from unittest.mock import MagicMock, patch

from mem0.memory.main import Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_instance():
    """Create a Memory instance with mocked internals (bypass __init__)."""
    with patch.object(Memory, "__init__", return_value=None):
        m = Memory.__new__(Memory)
        m.config = MagicMock()
        m.config.custom_fact_extraction_prompt = None
        m.config.custom_update_memory_prompt = None
        m.config.fallback_llm = None
        m.embedding_model = MagicMock()
        m.embedding_model.embed.return_value = [0.1] * 128
        m.llm = MagicMock()
        m._fallback_llm = None
        m.enable_graph = False
        m.graph = None
        m.db = MagicMock()
        m.vector_store = MagicMock()
        m.vector_store.search.return_value = []
        m.api_version = "v1.1"
        m.collection_name = "test_collection"
        return m


# ===========================================================================
# TestFilterSystemMessages
# ===========================================================================

class TestFilterSystemMessages:
    """Tests for system message filtering in _add_to_graph."""

    def test_system_messages_excluded_from_graph_data(self):
        """System messages should not be included in data sent to graph.add()."""
        m = _make_memory_instance()
        m.enable_graph = True
        m.graph = MagicMock()
        m.graph.add.return_value = {"added_entities": [], "deleted_entities": []}

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I like Python."},
            {"role": "assistant", "content": "Great choice!"},
        ]

        m._add_to_graph(messages, {"user_id": "u1"})

        # Verify graph.add was called with data that does NOT include system message
        data_arg = m.graph.add.call_args[0][0]
        assert "You are a helpful assistant" not in data_arg
        assert "I like Python" in data_arg
        assert "Great choice" in data_arg

    def test_only_system_messages_sends_empty_data(self):
        """If only system messages exist, graph data should be empty."""
        m = _make_memory_instance()
        m.enable_graph = True
        m.graph = MagicMock()
        m.graph.add.return_value = {"added_entities": [], "deleted_entities": []}

        messages = [
            {"role": "system", "content": "System prompt here."},
        ]

        m._add_to_graph(messages, {"user_id": "u1"})

        data_arg = m.graph.add.call_args[0][0]
        assert data_arg == ""

    def test_user_and_assistant_messages_preserved(self):
        """User and assistant messages should all be included."""
        m = _make_memory_instance()
        m.enable_graph = True
        m.graph = MagicMock()
        m.graph.add.return_value = {"added_entities": [], "deleted_entities": []}

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you"},
        ]

        m._add_to_graph(messages, {"user_id": "u1"})

        data_arg = m.graph.add.call_args[0][0]
        assert "Hello" in data_arg
        assert "Hi there" in data_arg
        assert "How are you" in data_arg

    def test_messages_without_content_key_skipped(self):
        """Messages missing 'content' key should be skipped gracefully."""
        m = _make_memory_instance()
        m.enable_graph = True
        m.graph = MagicMock()
        m.graph.add.return_value = {"added_entities": [], "deleted_entities": []}

        messages = [
            {"role": "user", "content": "Keep this"},
            {"role": "tool", "tool_call_id": "123"},  # no content
        ]

        m._add_to_graph(messages, {"user_id": "u1"})

        data_arg = m.graph.add.call_args[0][0]
        assert "Keep this" in data_arg

    def test_system_messages_filtered_in_infer_false_path(self):
        """In the non-infer path, system messages should be skipped for embedding."""
        m = _make_memory_instance()

        messages = [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": "User message"},
        ]

        result = m._add_to_vector_store(messages, {}, {"user_id": "u1"}, infer=False)

        # Only user message should have been processed (system skipped)
        assert len(result) == 1
        assert result[0]["memory"] == "User message"


# ===========================================================================
# TestFallbackLLMRetry
# ===========================================================================

class TestFallbackLLMRetry:
    """Tests for the 3-attempt retry with fallback LLM in fact extraction.

    The retry logic lives inside _add_to_vector_store(infer=True).
    We mock the LLM to test different failure/success scenarios.
    """

    def _make_memory_for_retry(self, primary_side_effect, fallback_response=None):
        """Set up Memory for retry testing via _add_to_vector_store.

        Reuses _make_memory_instance() and layers on retry-specific config.

        Args:
            primary_side_effect: list of return values / Exceptions for primary LLM
            fallback_response: optional string response for fallback LLM
        """
        m = _make_memory_instance()
        m.config.custom_fact_extraction_prompt = "Extract facts as JSON"
        m.llm.generate_response.side_effect = primary_side_effect

        if fallback_response is not None:
            fb_llm = MagicMock()
            fb_llm.generate_response.return_value = fallback_response
            m._fallback_llm = fb_llm

        return m

    def test_success_on_first_attempt(self):
        """Primary LLM succeeds on first try - 1 fact extraction call + 1 update call."""
        facts_json = json.dumps({"facts": ["User likes Python"]})
        # After fact extraction succeeds, there's also an update-memory LLM call
        update_resp = json.dumps({"memory": []})
        m = self._make_memory_for_retry([facts_json, update_resp])

        messages = [{"role": "user", "content": "I like Python"}]
        m._add_to_vector_store(messages, {}, {"user_id": "u1"}, infer=True)

        # 1 for fact extraction + 1 for update memory = 2
        assert m.llm.generate_response.call_count == 2

    def test_retry_on_first_failure(self):
        """First attempt fails, second succeeds."""
        facts_json = json.dumps({"facts": ["User likes Rust"]})
        update_resp = json.dumps({"memory": []})
        m = self._make_memory_for_retry([
            Exception("LLM timeout"),
            facts_json,
            update_resp,
        ])

        messages = [{"role": "user", "content": "I like Rust"}]
        m._add_to_vector_store(messages, {}, {"user_id": "u1"}, infer=True)

        # 1 fail + 1 success + 1 update = 3
        assert m.llm.generate_response.call_count == 3

    def test_fallback_on_third_attempt(self):
        """Primary fails twice, fallback LLM used on 3rd attempt."""
        fb_json = json.dumps({"facts": ["User likes Go"]})
        update_resp = json.dumps({"memory": []})
        m = self._make_memory_for_retry(
            # Primary fails twice, then called for update
            [Exception("fail 1"), Exception("fail 2"), update_resp],
            fallback_response=fb_json,
        )

        messages = [{"role": "user", "content": "I like Go"}]
        m._add_to_vector_store(messages, {}, {"user_id": "u1"}, infer=True)

        # Primary: 2 fails + 1 update = 3; Fallback: 1 fact extraction
        assert m.llm.generate_response.call_count == 3
        assert m._fallback_llm.generate_response.call_count == 1

    def test_no_fallback_all_primary(self):
        """Without fallback, all 3 attempts use primary LLM."""
        facts_json = json.dumps({"facts": ["data"]})
        update_resp = json.dumps({"memory": []})
        m = self._make_memory_for_retry([
            Exception("fail 1"),
            Exception("fail 2"),
            facts_json,
            update_resp,
        ])

        messages = [{"role": "user", "content": "data"}]
        m._add_to_vector_store(messages, {}, {"user_id": "u1"}, infer=True)

        # 2 fails + 1 success + 1 update = 4
        assert m.llm.generate_response.call_count == 4

    def test_all_attempts_fail_no_crash(self):
        """All 3 attempts fail - should not raise, returns empty results."""
        m = self._make_memory_for_retry([
            Exception("fail 1"),
            Exception("fail 2"),
            Exception("fail 3"),
        ])

        messages = [{"role": "user", "content": "data"}]
        # Should not raise
        m._add_to_vector_store(messages, {}, {"user_id": "u1"}, infer=True)

        assert m.llm.generate_response.call_count == 3
