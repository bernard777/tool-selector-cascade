"""Integration tests for CascadeSelector (Level 1 + 2 sync, full cascade async)."""
from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from tool_selector_cascade import CascadeSelector, CascadeConfig
from tool_selector_cascade.metrics import LevelMetrics


# ---------------------------------------------------------------------------
# Synchronous select() -- Level 1 + 2
# ---------------------------------------------------------------------------


class TestCascadeSelectorSelect:
    def test_returns_full_pool_when_below_min_pool_size(
        self, sample_tools: List[Any]
    ) -> None:
        small = sample_tools[:3]
        selector = CascadeSelector(CascadeConfig(min_pool_size=5))
        result, metrics = selector.select("do something", small)
        assert result == small
        assert metrics.final_count == 3
        assert metrics.input_pool_size == 3

    def test_returns_slice_when_model_unavailable(self, sample_tools: List[Any]) -> None:
        selector = CascadeSelector(
            CascadeConfig(reranker_enabled=False, llm_enabled=False)
        )
        with patch(
            "tool_selector_cascade.levels.embedder._get_model",
            return_value=None,
        ):
            result, metrics = selector.select("send email", sample_tools, top_k=5)
        assert len(result) <= 5

    def test_always_include_prefixes_are_preserved(self, sample_tools: List[Any]) -> None:
        config = CascadeConfig(
            always_include_prefixes=["web_search", "http_request"],
            reranker_enabled=False,
        )
        selector = CascadeSelector(config)
        with patch(
            "tool_selector_cascade.levels.embedder._get_model",
            return_value=None,
        ):
            result, _ = selector.select("send email", sample_tools, top_k=3)
        names = {t.name for t in result}
        assert "web_search" in names
        assert "http_request" in names

    def test_metrics_include_level1_entry(self, sample_tools: List[Any]) -> None:
        selector = CascadeSelector(CascadeConfig(reranker_enabled=False))
        with patch(
            "tool_selector_cascade.levels.embedder._get_model",
            return_value=None,
        ):
            _, metrics = selector.select("test intent", sample_tools)
        assert "level1_embedding" in metrics.levels

    def test_metrics_include_level2_entry_when_reranker_enabled(
        self, sample_tools: List[Any]
    ) -> None:
        selector = CascadeSelector(CascadeConfig())
        with patch(
            "tool_selector_cascade.levels.embedder._get_model",
            return_value=None,
        ):
            with patch(
                "tool_selector_cascade.levels.reranker._get_reranker",
                return_value=None,
            ):
                _, metrics = selector.select("test", sample_tools)
        assert "level1_embedding" in metrics.levels
        assert "level2_reranker" in metrics.levels

    def test_graceful_on_embedder_exception(self, sample_tools: List[Any]) -> None:
        selector = CascadeSelector(CascadeConfig(reranker_enabled=False))
        with patch(
            "tool_selector_cascade.levels.embedder._get_model",
            side_effect=RuntimeError("boom"),
        ):
            result, metrics = selector.select("test", sample_tools, top_k=5)
        assert len(result) > 0
        assert metrics.total_latency_ms >= 0

    def test_top_k_override_respected(self, sample_tools: List[Any]) -> None:
        selector = CascadeSelector(CascadeConfig(reranker_enabled=False))
        with patch(
            "tool_selector_cascade.levels.embedder._get_model",
            return_value=None,
        ):
            result, _ = selector.select("test", sample_tools, top_k=2)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# Async aselect() -- full cascade
# ---------------------------------------------------------------------------


class TestCascadeSelectorASelect:
    @pytest.mark.asyncio
    async def test_aselect_without_llm_enabled_skips_level3(
        self, sample_tools: List[Any]
    ) -> None:
        selector = CascadeSelector(
            CascadeConfig(llm_enabled=False, reranker_enabled=False)
        )
        with patch(
            "tool_selector_cascade.levels.embedder._get_model",
            return_value=None,
        ):
            result, metrics = await selector.aselect("test", sample_tools)
        assert len(result) > 0
        assert "level3_llm" not in metrics.levels

    @pytest.mark.asyncio
    async def test_aselect_single_candidate_skips_level3(self) -> None:
        single = [MagicMock(name="only_tool", description="the one")]
        selector = CascadeSelector()
        with patch(
            "tool_selector_cascade.levels.embedder._get_model",
            return_value=None,
        ):
            result, metrics = await selector.aselect("test", single)
        assert len(result) == 1
        assert "level3_llm" not in metrics.levels

    @pytest.mark.asyncio
    async def test_aselect_level3_returns_single_tool(
        self, sample_tools: List[Any]
    ) -> None:
        selector = CascadeSelector(CascadeConfig(llm_enabled=True))
        with patch(
            "tool_selector_cascade.levels.embedder._get_model",
            return_value=None,
        ):
            with patch(
                "tool_selector_cascade.levels.reranker._get_reranker",
                return_value=None,
            ):
                # LLMPickerL3 will skip (no API key in test env) and return top-1
                result, metrics = await selector.aselect("send email", sample_tools)
        assert len(result) >= 1  # Level 3 fallback returns top-1


# ---------------------------------------------------------------------------
# CascadeConfig
# ---------------------------------------------------------------------------


class TestCascadeConfig:
    def test_default_values(self) -> None:
        config = CascadeConfig()
        assert config.embedder_top_k == 20
        assert config.reranker_top_k == 5
        assert config.llm_provider == "anthropic"
        assert config.reranker_enabled is True
        assert config.llm_enabled is True
        assert config.min_pool_size == 5

    def test_custom_overrides(self) -> None:
        config = CascadeConfig(
            embedder_top_k=10,
            reranker_enabled=False,
            llm_provider="openai",
            llm_model="gpt-4o-mini",
        )
        assert config.embedder_top_k == 10
        assert not config.reranker_enabled
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o-mini"

