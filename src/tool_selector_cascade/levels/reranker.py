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
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from tool_selector_cascade.metrics import LevelMetrics, Timer
from tool_selector_cascade.types import tool_as_text

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_reranker: Any | None = None
_reranker_name_loaded: str | None = None
_reranker_lock = threading.Lock()
_reranker_attempted: bool = False


def _get_reranker(model_name: str) -> Any | None:
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
            from sentence_transformers import CrossEncoder

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
        forced_indices: list[int] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[T], LevelMetrics]:
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
            logger.warning(
                "RerankerL2: model unavailable -- returning candidates[:%d]",
                effective_top_k,
            )
            metrics.skipped = True
            metrics.output_count = min(n, effective_top_k)
            return list(tools[:effective_top_k]), metrics

        try:
            with Timer() as timer:
                pairs = [(intent, tool_as_text(t)) for t in tools]
                scores: Any = reranker.predict(pairs, show_progress_bar=False)

            metrics.latency_ms = timer.elapsed_ms

            _forced = list(forced_indices or [])
            remaining_slots = max(0, effective_top_k - len(_forced))

            candidates = [(i, float(scores[i])) for i in range(n) if i not in set(_forced)]
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
