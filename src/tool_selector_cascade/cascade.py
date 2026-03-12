"""CascadeSelector -- 3-level tool selection pipeline.

Orchestrates three successive levels that progressively narrow a large tool
pool down to the single most relevant tool for a given user intent.

::

    [1000 tools]
         |
         v  Level 1 -- Embedding  (~20 ms, local, ~$0)
         |  intfloat/multilingual-e5-base
         |
    [top 20]
         |
         v  Level 2 -- Reranker   (~243 ms, local, ~$0)
         |  Alibaba-NLP/gte-reranker-modernbert-base
         |
    [top 5]
         |
         v  Level 3 -- Micro-LLM  (~150 ms, API, ~$0.0001)
         |  claude-haiku-4-5 / gemini-2.5-flash-lite / gpt-4o-mini
         |
    [1 tool]

Each level degrades gracefully: if a model is unavailable or an API call fails,
the cascade continues with the best result from the previous level.

Example
-------
.. code-block:: python

    from tool_selector_cascade import CascadeSelector

    selector = CascadeSelector()
    selector.warm_up(all_tools)  # non-blocking in a background thread

    # Sync -- Level 1 + 2 only (no LLM cost)
    candidates, metrics = selector.select(intent, tools, top_k=5)

    # Async -- full 3-level cascade
    result, metrics = await selector.aselect(intent, tools)
    best_tool = result[0]
    print(metrics.as_dict())
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from tool_selector_cascade.config import CascadeConfig
from tool_selector_cascade.levels.embedder import EmbedderL1
from tool_selector_cascade.levels.llm_picker import LLMPickerL3
from tool_selector_cascade.levels.reranker import RerankerL2
from tool_selector_cascade.metrics import SelectionMetrics
from tool_selector_cascade.types import extract_tool_name

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CascadeSelector(Generic[T]):
    """3-level cascading tool selector.

    Parameters
    ----------
    config:
        Full cascade configuration.  Uses :class:`~tool_selector_cascade.config.CascadeConfig`
        defaults if not provided.
    """

    def __init__(self, config: CascadeConfig | None = None) -> None:
        self.config: CascadeConfig = config or CascadeConfig()
        self._embedder: EmbedderL1[Any] = EmbedderL1(
            model_name=self.config.embedder_model,
            top_k=self.config.embedder_top_k,
            min_pool=self.config.embedder_min_pool,
        )
        self._reranker: RerankerL2[Any] = RerankerL2(
            model_name=self.config.reranker_model,
            top_k=self.config.reranker_top_k,
        )
        self._llm_picker: LLMPickerL3[Any] = LLMPickerL3(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
            timeout=self.config.llm_timeout,
        )

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def warm_up(self, tools: Sequence[Any] | None = None) -> bool:
        """Pre-load all local models (Level 1 + Level 2).

        Calling this once at application startup in a background thread
        eliminates cold-start latency on the first real query.

        Parameters
        ----------
        tools:
            Optional pool to pre-encode via Level 1.  If provided, the first
            :meth:`select` call on this exact pool is a cache hit.

        Returns
        -------
        bool
            ``True`` if the Level 1 embedding model loaded successfully.
        """
        ok = self._embedder.warm_up(tools)
        if self.config.reranker_enabled:
            self._reranker.warm_up()
        return ok

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _forced_indices(self, tools: Sequence[Any]) -> list[int]:
        """Return indices of tools whose names match *always_include_prefixes*."""
        if not self.config.always_include_prefixes:
            return []
        return [
            i
            for i, t in enumerate(tools)
            if any(extract_tool_name(t).startswith(p) for p in self.config.always_include_prefixes)
        ]

    # ------------------------------------------------------------------
    # Synchronous API (Level 1 + 2)
    # ------------------------------------------------------------------

    def select(
        self,
        intent: str,
        tools: Sequence[T],
        *,
        top_k: int | None = None,
    ) -> tuple[list[T], SelectionMetrics]:
        """Run Level 1 (embedding) and Level 2 (reranker) synchronously.

        Level 3 (LLM) is NOT invoked here.  Use :meth:`aselect` for the full
        cascade.

        Parameters
        ----------
        intent:
            User task description.
        tools:
            Full tool pool.
        top_k:
            Override ``config.reranker_top_k`` for this single call.

        Returns
        -------
        tuple[List[T], SelectionMetrics]
            Filtered and ranked tool list plus cascade metrics.
        """
        n = len(tools)
        sm = SelectionMetrics(input_pool_size=n)

        if n <= self.config.min_pool_size:
            sm.final_count = n
            return list(tools), sm

        forced = self._forced_indices(tools)

        # ── Level 1: Embedding ───────────────────────────────────────────
        l1_result, l1_metrics = self._embedder.filter(intent, tools, forced_indices=forced)
        sm.add_level("level1_embedding", l1_metrics)

        if not self.config.reranker_enabled:
            effective_k = top_k if top_k is not None else self.config.reranker_top_k
            final = l1_result[:effective_k]
            sm.final_count = len(final)
            return final, sm

        # ── Level 2: Reranker ────────────────────────────────────────────
        l2_forced = self._forced_indices(l1_result)

        l2_result, l2_metrics = self._reranker.rerank(
            intent, l1_result, forced_indices=l2_forced, top_k=top_k
        )

        sm.add_level("level2_reranker", l2_metrics)
        sm.final_count = len(l2_result)

        return l2_result, sm

    # ------------------------------------------------------------------
    # Async API (all 3 levels)
    # ------------------------------------------------------------------

    async def aselect(
        self,
        intent: str,
        tools: Sequence[T],
    ) -> tuple[list[T], SelectionMetrics]:
        """Run the full 3-level cascade asynchronously.

        Parameters
        ----------
        intent:
            User task description.
        tools:
            Full tool pool.

        Returns
        -------
        tuple[List[T], SelectionMetrics]
            When Level 3 succeeds, ``result[0]`` is the single best tool.
            On Level 3 failure, returns the top Level 2 candidate.
        """
        l2_result, sm = await asyncio.to_thread(self.select, intent, tools)

        if not self.config.llm_enabled or len(l2_result) <= 1:
            return l2_result, sm

        # ── Level 3: Micro-LLM ──────────────────────────────────────────
        l3_result, l3_metrics = await self._llm_picker.pick(intent, l2_result)
        sm.add_level("level3_llm", l3_metrics)
        sm.final_count = len(l3_result)

        return l3_result, sm
