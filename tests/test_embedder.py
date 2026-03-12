"""Unit tests for the Level 1 embedding filter (EmbedderL1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

import tool_selector_cascade.levels.embedder as _emb_mod
from tool_selector_cascade.levels.embedder import EmbedderL1, _pool_cache_key


def _make_tool(name: str, description: str = "") -> MagicMock:
    t = MagicMock(spec_set=["name", "description"])
    t.name = name
    t.description = description
    return t


# ---------------------------------------------------------------------------
# filter() -- basic behaviour
# ---------------------------------------------------------------------------


class TestEmbedderL1Filter:
    def test_returns_full_pool_when_below_min_pool(self) -> None:
        embedder = EmbedderL1(min_pool=10, top_k=5)
        tools = [_make_tool(f"tool_{i}") for i in range(5)]
        result, metrics = embedder.filter("intent", tools)
        assert result == tools
        assert metrics.skipped is True
        assert metrics.input_count == 5
        assert metrics.output_count == 5

    def test_fallback_when_model_unavailable(self) -> None:
        embedder = EmbedderL1(top_k=5, min_pool=3)
        tools = [_make_tool(f"tool_{i}") for i in range(10)]
        with patch("tool_selector_cascade.levels.embedder._get_model", return_value=None):
            result, metrics = embedder.filter("test", tools)
        assert len(result) <= 5
        assert metrics.skipped is True

    def test_cache_avoids_re_encoding_same_pool(self) -> None:
        _emb_mod._embedding_cache.clear()
        embedder = EmbedderL1(top_k=3, min_pool=3)
        tools = [_make_tool(f"t_{i}", f"desc {i}") for i in range(10)]

        mock_model = MagicMock()
        # Return valid normalised embeddings of shape (N, D) or (1, D)
        mock_model.encode.side_effect = [
            np.random.rand(10, 64).astype(np.float32),  # pool encoding
            np.random.rand(1, 64).astype(np.float32),  # intent encoding -- first call
            np.random.rand(1, 64).astype(np.float32),  # intent encoding -- second call
        ]

        with patch("tool_selector_cascade.levels.embedder._get_model", return_value=mock_model):
            embedder.filter("first call", tools)
            first_encode_count = mock_model.encode.call_count  # should be 2 (pool + intent)
            embedder.filter("second call", tools)  # same pool -> cache hit

        # Only one extra encode call (the new intent), not a full re-encode of the pool
        assert mock_model.encode.call_count == first_encode_count + 1

    def test_forced_indices_always_in_result(self) -> None:
        _emb_mod._embedding_cache.clear()
        embedder = EmbedderL1(top_k=3, min_pool=3)
        n = 10
        tools = [_make_tool(f"t_{i}") for i in range(n)]
        forced = [7, 9]

        pool_emb = np.zeros((n, 8), dtype=np.float32)
        intent_emb = np.ones((1, 8), dtype=np.float32)
        # Give forced tools a VERY LOW score to prove they are included regardless
        pool_emb[7] = np.zeros(8, dtype=np.float32) - 0.9
        pool_emb[9] = np.zeros(8, dtype=np.float32) - 0.9

        mock_model = MagicMock()
        mock_model.encode.side_effect = [pool_emb, intent_emb]  # shape (1,8) so [0] gives (8,)

        with patch("tool_selector_cascade.levels.embedder._get_model", return_value=mock_model):
            result, _ = embedder.filter("intent", tools, forced_indices=forced)

        result_indices = [tools.index(t) for t in result]
        for fi in forced:
            assert fi in result_indices

    def test_output_count_matches_result_length(self) -> None:
        _emb_mod._embedding_cache.clear()
        embedder = EmbedderL1(top_k=4, min_pool=3)
        tools = [_make_tool(f"t_{i}") for i in range(10)]

        pool_emb = np.random.rand(10, 8).astype(np.float32)
        pool_emb /= np.linalg.norm(pool_emb, axis=1, keepdims=True)
        intent_emb = np.random.rand(8).astype(np.float32)
        intent_emb /= np.linalg.norm(intent_emb)

        mock_model = MagicMock()
        mock_model.encode.side_effect = [pool_emb, intent_emb]

        with patch("tool_selector_cascade.levels.embedder._get_model", return_value=mock_model):
            result, metrics = embedder.filter("intent", tools)

        assert metrics.output_count == len(result)
        assert len(result) <= 4


# ---------------------------------------------------------------------------
# _pool_cache_key
# ---------------------------------------------------------------------------


class TestPoolCacheKey:
    def test_same_texts_same_key(self) -> None:
        texts = ["tool_a: do something", "tool_b: do other thing"]
        assert _pool_cache_key(texts) == _pool_cache_key(texts)

    def test_different_texts_different_key(self) -> None:
        assert _pool_cache_key(["a", "b"]) != _pool_cache_key(["a", "c"])

    def test_order_matters(self) -> None:
        assert _pool_cache_key(["a", "b"]) != _pool_cache_key(["b", "a"])

    def test_empty_list(self) -> None:
        key = _pool_cache_key([])
        assert isinstance(key, str) and len(key) == 32  # MD5 hex


# ---------------------------------------------------------------------------
# warm_up()
# ---------------------------------------------------------------------------


class TestWarmUp:
    def test_warm_up_returns_false_when_model_unavailable(self) -> None:
        embedder = EmbedderL1()
        with patch("tool_selector_cascade.levels.embedder._get_model", return_value=None):
            result = embedder.warm_up()
        assert result is False

    def test_warm_up_populates_cache(self) -> None:
        _emb_mod._embedding_cache.clear()
        embedder = EmbedderL1()
        tools = [_make_tool(f"t_{i}") for i in range(5)]

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((5, 8), dtype=np.float32)

        with patch("tool_selector_cascade.levels.embedder._get_model", return_value=mock_model):
            result = embedder.warm_up(tools=tools)

        assert result is True
        assert len(_emb_mod._embedding_cache) > 0
