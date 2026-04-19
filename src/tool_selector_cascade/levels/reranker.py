"""Level 2 -- Cross-encoder reranker.

Re-scores the top-20 candidates from Level 1 using a cross-encoder model
that jointly attends to both the intent and the tool text.  This is
computationally heavier than a bi-encoder but yields higher-quality rankings.

Model profile
-------------
- HuggingFace ID  : ``Alibaba-NLP/gte-reranker-modernbert-base``
- Parameters      : 149 M
- RAM footprint   : ~300 MB
- Latency         : ~243 ms / 100 pairs (warm, CPU)
- Cost            : $0 (local inference)

Graceful degradation
--------------------
If ``sentence-transformers`` is not installed or the model is unavailable,
the level returns the first ``top_k`` candidates from Level 1 unchanged.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

from tool_selector_cascade.metrics import LevelMetrics, Timer
from tool_selector_cascade import tool_category_boost
from tool_selector_cascade.types import tool_as_text

logger = logging.getLogger(__name__)

T = TypeVar("T")

INFRASTRUCTURE_FIXED_SCORE = 1.0


def _normalize_category(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.strip().lower()


def _extract_tool_category(tool: Any) -> str:
    """Best-effort extraction of a tool category from duck-typed tool payloads."""
    # Object attributes
    for attr in ("category", "tool_category"):
        value = _normalize_category(getattr(tool, attr, None))
        if value:
            return value

    # Dict-like payloads (tool dict or OpenAI-like schema wrappers)
    if isinstance(tool, dict):
        for key in ("category", "tool_category"):
            value = _normalize_category(tool.get(key))
            if value:
                return value

        metadata = tool.get("metadata")
        if isinstance(metadata, dict):
            for key in ("category", "tool_category"):
                value = _normalize_category(metadata.get(key))
                if value:
                    return value

        function_payload = tool.get("function")
        if isinstance(function_payload, dict):
            value = _normalize_category(function_payload.get("category") or function_payload.get("tool_category"))
            if value:
                return value

    return ""

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_reranker: Optional[Any] = None
_reranker_name_loaded: Optional[str] = None
_reranker_lock = threading.Lock()
_reranker_attempted: bool = False


def _get_reranker(model_name: str) -> Optional[Any]:
    """Return the CrossEncoder singleton, loading it on first call.

    Thread-safe.  Returns ``None`` on load failure and does not retry.
    """
    global _reranker, _reranker_name_loaded, _reranker_attempted

    with _reranker_lock:
        if _reranker is not None and _reranker_name_loaded == model_name:
            return _reranker
        if _reranker_attempted and _reranker_name_loaded == model_name:
            return None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            logger.info("RerankerL2: loading model '%s' ...", model_name)
            _reranker = CrossEncoder(model_name)
            _reranker_name_loaded = model_name
            logger.info("RerankerL2: model ready")
        except Exception as exc:
            logger.warning(
                "RerankerL2: could not load model '%s': %s -- falling back",
                model_name,
                exc,
            )
            _reranker = None
            _reranker_name_loaded = model_name
        finally:
            _reranker_attempted = True
    return _reranker


class RerankerL2(Generic[T]):
    """Level-2 cross-encoder reranker.

    Re-scores candidates from Level 1 using cross-attention between the
    user intent and each candidate tool text.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
        Defaults to ``Alibaba-NLP/gte-reranker-modernbert-base``.
    top_k:
        Number of candidates to forward to Level 3. Default: 5.
    """

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-reranker-modernbert-base",
        top_k: int = 5,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k

    def warm_up(self) -> bool:
        """Pre-load the reranker model.

        Returns
        -------
        bool
            ``True`` if the model loaded successfully.
        """
        return _get_reranker(self.model_name) is not None

    def rerank(
        self,
        intent: str,
        tools: Sequence[T],
        *,
        forced_indices: Optional[List[int]] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[T], LevelMetrics]:
        """Re-score *tools* by cross-encoder relevance to *intent*.

        Parameters
        ----------
        intent:
            User task description.
        tools:
            Candidates from Level 1 (typically top-20).
        forced_indices:
            Indices of tools guaranteed to appear in the output.
        top_k:
            Per-call override for the number of results to return.
            If ``None`` (default), ``self.top_k`` is used.

        Returns
        -------
        tuple[List[T], LevelMetrics]
        """
        effective_top_k = top_k if top_k is not None else self.top_k
        metrics = LevelMetrics(input_count=len(tools))
        n = len(tools)

        if n == 0:
            return [], metrics

        reranker = _get_reranker(self.model_name)
        if reranker is None:
            logger.warning("RerankerL2: model unavailable -- returning candidates[:%d]", effective_top_k)
            metrics.skipped = True
            metrics.output_count = min(n, effective_top_k)
            return list(tools[:effective_top_k]), metrics

        try:
            with Timer() as timer:
                categories = [_extract_tool_category(tool) for tool in tools]
                infra_categories = {_normalize_category(value) for value in tool_category_boost.infrastructure}
                category_weights = {
                    _normalize_category(key): float(value)
                    for key, value in getattr(tool_category_boost, "weights", {}).items()
                }

                semantic_indices = [
                    index
                    for index, category in enumerate(categories)
                    if category not in infra_categories
                ]

                score_by_index: Dict[int, float] = {}
                if semantic_indices:
                    pairs = [(intent, tool_as_text(tools[index])) for index in semantic_indices]
                    semantic_scores: Any = reranker.predict(pairs, show_progress_bar=False)
                    for local_rank, tool_index in enumerate(semantic_indices):
                        category = categories[tool_index]
                        score_by_index[tool_index] = float(semantic_scores[local_rank]) + category_weights.get(category, 0.0)

                infra_indices = [
                    index
                    for index, category in enumerate(categories)
                    if category in infra_categories
                ]
                for tool_index in infra_indices:
                    score_by_index[tool_index] = INFRASTRUCTURE_FIXED_SCORE

            metrics.latency_ms = timer.elapsed_ms

            _forced = list(forced_indices or [])
            remaining_slots = max(0, effective_top_k - len(_forced))

            candidates = [
                (i, score_by_index.get(i, INFRASTRUCTURE_FIXED_SCORE)) for i in range(n) if i not in set(_forced)
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_indices = _forced + [i for i, _ in candidates[:remaining_slots]]

            result = [tools[i] for i in top_indices]
            metrics.output_count = len(result)

            logger.info(
                "RerankerL2: %d/%d candidates re-ranked in %.1f ms | top: %s",
                len(result),
                n,
                metrics.latency_ms,
                [str(getattr(t, "name", t)) for t in result[:3]],
            )
            return result, metrics

        except Exception as exc:
            logger.warning("RerankerL2: unexpected error '%s' -- fallback", exc)
            metrics.error = str(exc)
            metrics.skipped = True
            metrics.output_count = min(n, effective_top_k)
            return list(tools[:effective_top_k]), metrics

