"""Unit tests for mem0.reranker.tei_reranker (TEI HTTP-based reranker)."""

from unittest.mock import MagicMock, patch

import pytest

from mem0.reranker.tei_reranker import TEIReranker
from mem0.configs.rerankers.tei import TEIRerankerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reranker(base_url="http://localhost:8184", top_k=None, timeout=10):
    """Create a TEIReranker with a TEIRerankerConfig."""
    config = TEIRerankerConfig(base_url=base_url, top_k=top_k, timeout=timeout)
    return TEIReranker(config)


def _mock_response(scores_by_index):
    """Build a mock requests.Response with TEI-style JSON body.

    Args:
        scores_by_index: list of (index, score) tuples
    """
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = [
        {"index": idx, "score": score} for idx, score in scores_by_index
    ]
    return resp


# ===========================================================================
# TestInit
# ===========================================================================

class TestInit:
    """Tests for TEIReranker initialization."""

    def test_init_with_tei_config(self):
        """Should accept a TEIRerankerConfig directly."""
        config = TEIRerankerConfig(base_url="http://gpu:8184", top_k=5, timeout=15)
        reranker = TEIReranker(config)
        assert reranker.rerank_url == "http://gpu:8184/rerank"
        assert reranker.config.top_k == 5
        assert reranker.config.timeout == 15

    def test_init_with_dict_config(self):
        """Should accept a dict and convert to TEIRerankerConfig."""
        reranker = TEIReranker({"base_url": "http://x:9999", "top_k": 3})
        assert reranker.rerank_url == "http://x:9999/rerank"
        assert reranker.config.top_k == 3

    def test_init_strips_trailing_slash(self):
        """base_url trailing slash should be stripped before appending /rerank."""
        reranker = _make_reranker(base_url="http://host:8184/")
        assert reranker.rerank_url == "http://host:8184/rerank"


# ===========================================================================
# TestRerank
# ===========================================================================

class TestRerank:
    """Tests for the rerank() method."""

    @patch("mem0.reranker.tei_reranker.requests.post")
    def test_basic_rerank(self, mock_post):
        """Should call TEI /rerank and return documents sorted by score."""
        mock_post.return_value = _mock_response([(0, 0.3), (1, 0.9)])

        reranker = _make_reranker()
        docs = [
            {"memory": "Python is great"},
            {"memory": "Rust is fast"},
        ]
        result = reranker.rerank("best language", docs)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["query"] == "best language"
        assert call_kwargs[1]["json"]["texts"] == ["Python is great", "Rust is fast"]

        # Sorted descending by score
        assert len(result) == 2
        assert result[0]["memory"] == "Rust is fast"
        assert result[0]["rerank_score"] == 0.9
        assert result[1]["memory"] == "Python is great"
        assert result[1]["rerank_score"] == 0.3

    @patch("mem0.reranker.tei_reranker.requests.post")
    def test_top_k_limits_results(self, mock_post):
        """top_k parameter should limit the number of returned documents."""
        mock_post.return_value = _mock_response([(0, 0.1), (1, 0.9), (2, 0.5)])

        reranker = _make_reranker()
        docs = [
            {"memory": "A"},
            {"memory": "B"},
            {"memory": "C"},
        ]
        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        assert result[0]["rerank_score"] == 0.9

    @patch("mem0.reranker.tei_reranker.requests.post")
    def test_config_top_k_applied(self, mock_post):
        """top_k from config should be used when not passed as parameter."""
        mock_post.return_value = _mock_response([(0, 0.1), (1, 0.9), (2, 0.5)])

        reranker = _make_reranker(top_k=1)
        docs = [{"memory": "A"}, {"memory": "B"}, {"memory": "C"}]
        result = reranker.rerank("query", docs)

        assert len(result) == 1

    @patch("mem0.reranker.tei_reranker.requests.post")
    def test_fallback_on_http_failure(self, mock_post):
        """When HTTP request fails, should return original order with score 0.0."""
        mock_post.side_effect = Exception("Connection refused")

        reranker = _make_reranker()
        docs = [
            {"memory": "A"},
            {"memory": "B"},
        ]
        result = reranker.rerank("query", docs)

        assert len(result) == 2
        assert result[0]["memory"] == "A"
        assert result[0]["rerank_score"] == 0.0
        assert result[1]["memory"] == "B"

    @patch("mem0.reranker.tei_reranker.requests.post")
    def test_fallback_respects_top_k(self, mock_post):
        """Fallback path should also respect top_k."""
        mock_post.side_effect = Exception("timeout")

        reranker = _make_reranker()
        docs = [{"memory": "A"}, {"memory": "B"}, {"memory": "C"}]
        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2

    def test_empty_documents(self):
        """Empty documents list should be returned immediately."""
        reranker = _make_reranker()
        result = reranker.rerank("query", [])
        assert result == []

    @patch("mem0.reranker.tei_reranker.requests.post")
    def test_extracts_text_from_various_keys(self, mock_post):
        """Should extract text from 'memory', 'text', or 'content' keys."""
        mock_post.return_value = _mock_response([(0, 0.5), (1, 0.8), (2, 0.3)])

        reranker = _make_reranker()
        docs = [
            {"memory": "from memory"},
            {"text": "from text"},
            {"content": "from content"},
        ]
        result = reranker.rerank("query", docs)

        texts_sent = mock_post.call_args[1]["json"]["texts"]
        assert texts_sent == ["from memory", "from text", "from content"]

    @patch("mem0.reranker.tei_reranker.requests.post")
    def test_preserves_original_doc_fields(self, mock_post):
        """Reranked docs should retain all original fields plus rerank_score."""
        mock_post.return_value = _mock_response([(0, 0.9)])

        reranker = _make_reranker()
        docs = [{"memory": "hello", "id": "abc", "extra": 42}]
        result = reranker.rerank("query", docs)

        assert result[0]["id"] == "abc"
        assert result[0]["extra"] == 42
        assert result[0]["rerank_score"] == 0.9

    @patch("mem0.reranker.tei_reranker.requests.post")
    def test_timeout_passed_to_request(self, mock_post):
        """HTTP timeout from config should be passed to requests.post."""
        mock_post.return_value = _mock_response([(0, 0.5)])

        reranker = _make_reranker(timeout=30)
        reranker.rerank("q", [{"memory": "x"}])

        assert mock_post.call_args[1]["timeout"] == 30
