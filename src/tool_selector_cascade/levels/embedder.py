"""Level 1 -- Bi-encoder embedding filter.

Uses ``intfloat/multilingual-e5-base`` to produce sentence embeddings for both
the user intent and all tool text representations (``"name: description"``).
Tool embeddings are cached by a stable MD5 hash of the tool pool so repeated
calls on the same pool are near-instant.

Model profile
-------------
- HuggingFace ID  : ``intfloat/multilingual-e5-base``
- Parameters      : 278 M
- RAM footprint   : ~270 MB
- Languages       : 100+ (FR, EN, DE, ES, ZH, ...)
- Latency         : ~20 ms per query (warm, CPU)
- Cost            : $0 (local inference)

Graceful degradation
--------------------
If ``sentence-transformers`` is not installed or the model fails to load, the
filter silently returns the first ``top_k`` tools from the original pool.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

from tool_selector_cascade.metrics import LevelMetrics, Timer
from tool_selector_cascade.types import tool_as_text

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Model singleton -- one per (model_name, process) pair, lazy-loaded
# ---------------------------------------------------------------------------
_model: Optional[Any] = None
_model_name_loaded: Optional[str] = None
_model_lock = threading.Lock()
_model_attempted: bool = False

# Per-pool embedding cache:  MD5(tool_texts) -> numpy array (N, D)
# Bounded to _MAX_CACHE_SIZE entries with FIFO eviction (dict keeps insertion order, Python 3.7+).
_MAX_CACHE_SIZE: int = 32
_cache_lock = threading.Lock()
_embedding_cache: Dict[str, Any] = {}


def _pool_cache_key(texts: List[str]) -> str:
    """Return a stable hex digest for a list of tool text strings."""
    h = hashlib.md5()
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _get_model(model_name: str) -> Optional[Any]:
    """Return the SentenceTransformer singleton, loading it on first call.

    Thread-safe.  Returns ``None`` if the model cannot be loaded; will NOT
    attempt to reload on subsequent calls (avoids repeated slow failures).
    """
    global _model, _model_name_loaded, _model_attempted

    with _model_lock:
        if _model is not None and _model_name_loaded == model_name:
            return _model
        if _model_attempted and _model_name_loaded == model_name:
            return None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info("EmbedderL1: loading model '%s' ...", model_name)
            _model = SentenceTransformer(model_name)
            _model_name_loaded = model_name
            logger.info("EmbedderL1: model ready")
        except Exception as exc:
            logger.warning(
                "EmbedderL1: could not load model '%s': %s -- falling back to slice",
                model_name,
                exc,
            )
            _model = None
            _model_name_loaded = model_name
        finally:
            _model_attempted = True
    return _model


class EmbedderL1(Generic[T]):
    """Level-1 bi-encoder filter that uses cosine similarity to narrow a tool pool.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
        Defaults to ``intfloat/multilingual-e5-base``.
    top_k:
        Number of candidates to return. Default: 20.
    min_pool:
        If ``len(tools) <= min_pool`` the full pool is returned unchanged.
        Default: 20.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        top_k: int = 20,
        min_pool: int = 20,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self.min_pool = min_pool

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def warm_up(self, tools: Optional[Sequence[Any]] = None) -> bool:
        """Pre-load the model and optionally pre-encode a tool pool.

        Call this once during application startup (in a background thread
        if you want non-blocking behaviour) to avoid cold-start latency on
        the first real query.

        Parameters
        ----------
        tools:
            Optional pool to pre-encode.  The embeddings are stored in the
            module-level cache so the next :meth:`filter` call on this pool
            is a cache hit (near-instant).

        Returns
        -------
        bool
            ``True`` if the model loaded successfully.
        """
        model = _get_model(self.model_name)
        if model is None:
            return False
        if tools:
            try:
                texts = [tool_as_text(t) for t in tools]
                key = _pool_cache_key(texts)
                if key not in _embedding_cache:
                    logger.debug(
                        "EmbedderL1 warm_up: pre-encoding %d tools into cache", len(texts)
                    )
                    embs: Any = model.encode(
                        texts,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=64,
                    )
                    with _cache_lock:
                        if key not in _embedding_cache:
                            if len(_embedding_cache) >= _MAX_CACHE_SIZE:
                                del _embedding_cache[next(iter(_embedding_cache))]
                            _embedding_cache[key] = embs
            except Exception as exc:
                logger.warning("EmbedderL1 warm_up: pre-encoding failed (non-blocking): %s", exc)
        return True

    # ------------------------------------------------------------------
    # Core filter
    # ------------------------------------------------------------------

    def filter(
        self,
        intent: str,
        tools: Sequence[T],
        *,
        forced_indices: Optional[List[int]] = None,
    ) -> Tuple[List[T], LevelMetrics]:
        """Filter *tools* by cosine similarity to *intent*.

        Parameters
        ----------
        intent:
            User task description (natural language).
        tools:
            Full tool pool to filter.
        forced_indices:
            Indices of tools that are guaranteed to appear in the output
            regardless of their similarity score.

        Returns
        -------
        tuple[List[T], LevelMetrics]
            A ranked subset of *tools* (length <= ``top_k``) and the
            corresponding level metrics.
        """
        metrics = LevelMetrics(input_count=len(tools))
        n = len(tools)

        # Fast path: pool small enough to skip filtering
        if n <= self.min_pool:
            metrics.skipped = True
            metrics.output_count = n
            return list(tools), metrics

        try:
            model = _get_model(self.model_name)
            if model is None:
                logger.warning("EmbedderL1: model unavailable -- returning tools[:%d]", self.top_k)
                metrics.skipped = True
                _forced = list(forced_indices or [])
                if _forced:
                    forced_set = set(_forced)
                    rest = [tools[i] for i in range(n) if i not in forced_set]
                    remaining = max(0, self.top_k - len(_forced))
                    result = [tools[i] for i in _forced] + rest[:remaining]
                else:
                    result = list(tools[: self.top_k])
                metrics.output_count = len(result)
                return result, metrics

        except Exception as exc:
            logger.warning("EmbedderL1: unexpected error '%s' -- fallback", exc)
            metrics.error = str(exc)
            metrics.skipped = True
            metrics.output_count = min(n, self.top_k)
            return list(tools[: self.top_k]), metrics

        try:
            with Timer() as timer:
                # Build or retrieve cached tool embeddings
                tool_texts = [tool_as_text(t) for t in tools]
                key = _pool_cache_key(tool_texts)
                if key not in _embedding_cache:
                    logger.debug("EmbedderL1: encoding %d tool descriptions (cache miss)", n)
                    pool_embs: Any = model.encode(
                        tool_texts,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=64,
                    )
                    with _cache_lock:
                        if key not in _embedding_cache:
                            if len(_embedding_cache) >= _MAX_CACHE_SIZE:
                                del _embedding_cache[next(iter(_embedding_cache))]
                            _embedding_cache[key] = pool_embs
                tool_embs: Any = _embedding_cache[key]

                # Encode user intent
                intent_emb: Any = model.encode(
                    [intent],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]

                # Cosine similarity (dot product on L2-normalised vectors)
                scores: Any = tool_embs @ intent_emb  # shape (N,)

            metrics.latency_ms = timer.elapsed_ms

            # Apply forced-include logic
            _forced = list(forced_indices or [])
            remaining_slots = max(0, self.top_k - len(_forced))
            if remaining_slots <= 0:
                result = [tools[i] for i in _forced[: self.top_k]]
                metrics.output_count = len(result)
                return result, metrics

            candidates = [i for i in range(n) if i not in set(_forced)]
            ranked = sorted(candidates, key=lambda i: float(scores[i]), reverse=True)
            top_indices = _forced + ranked[:remaining_slots]

            result = [tools[i] for i in top_indices]
            metrics.output_count = len(result)

            logger.info(
                "EmbedderL1: %d/%d tools selected in %.1f ms | top: %s",
                len(result),
                n,
                metrics.latency_ms,
                [str(getattr(t, "name", t)) for t in result[:3]],
            )
            return result, metrics

        except Exception as exc:
            logger.warning("EmbedderL1: unexpected error '%s' -- fallback", exc)
            metrics.error = str(exc)
            metrics.skipped = True
            metrics.output_count = min(n, self.top_k)
            return list(tools[: self.top_k]), metrics

