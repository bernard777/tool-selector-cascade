"""Unit tests for the Level 2 cross-encoder reranker (RerankerL2)."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tool_selector_cascade.levels.reranker import RerankerL2


def _make_tool(name: str, description: str = "") -> MagicMock:
    t = MagicMock(spec_set=["name", "description"])
    t.name = name
    t.description = description
    return t


class TestRerankerL2Rerank:
    def test_returns_empty_for_empty_tools(self) -> None:
        reranker = RerankerL2()
        with patch(
            "tool_selector_cascade.levels.reranker._get_reranker", return_value=None
        ):
            result, metrics = reranker.rerank("intent", [])
        assert result == []
        assert metrics.input_count == 0

    def test_fallback_when_model_unavailable(self) -> None:
        reranker = RerankerL2(top_k=3)
        tools = [_make_tool(f"t_{i}") for i in range(8)]
        with patch(
            "tool_selector_cascade.levels.reranker._get_reranker", return_value=None
        ):
            result, metrics = reranker.rerank("test", tools)
        assert len(result) <= 3
        assert metrics.skipped is True

    def test_reranks_by_score(self) -> None:
        reranker = RerankerL2(top_k=3)
        tools = [_make_tool(f"t_{i}") for i in range(5)]
        # Scores: t_4 best, t_0 worst
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = scores

        with patch(
            "tool_selector_cascade.levels.reranker._get_reranker",
            return_value=mock_reranker,
        ):
            result, metrics = reranker.rerank("intent", tools)

        assert result[0].name == "t_4"  # highest score first
        assert len(result) == 3
        assert metrics.output_count == 3
        assert metrics.latency_ms >= 0
        assert not metrics.skipped

    def test_forced_indices_always_in_result(self) -> None:
        reranker = RerankerL2(top_k=3)
        tools = [_make_tool(f"t_{i}") for i in range(6)]
        forced = [5]  # t_5 has a low score
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1], dtype=np.float32)

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = scores

        with patch(
            "tool_selector_cascade.levels.reranker._get_reranker",
            return_value=mock_reranker,
        ):
            result, _ = reranker.rerank("intent", tools, forced_indices=forced)

        result_names = [t.name for t in result]
        assert "t_5" in result_names

    def test_output_count_equals_result_length(self) -> None:
        reranker = RerankerL2(top_k=2)
        tools = [_make_tool(f"t_{i}") for i in range(5)]
        scores = np.array([0.5, 0.3, 0.9, 0.1, 0.7], dtype=np.float32)

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = scores

        with patch(
            "tool_selector_cascade.levels.reranker._get_reranker",
            return_value=mock_reranker,
        ):
            result, metrics = reranker.rerank("intent", tools)

        assert metrics.output_count == len(result) == 2

    def test_gracefully_handles_predict_exception(self) -> None:
        reranker = RerankerL2(top_k=2)
        tools = [_make_tool(f"t_{i}") for i in range(5)]

        mock_reranker = MagicMock()
        mock_reranker.predict.side_effect = RuntimeError("GPU OOM")

        with patch(
            "tool_selector_cascade.levels.reranker._get_reranker",
            return_value=mock_reranker,
        ):
            result, metrics = reranker.rerank("intent", tools)

        assert len(result) > 0
        assert metrics.error is not None

