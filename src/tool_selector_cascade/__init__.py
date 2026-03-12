"""tool-selector-cascade
~~~~~~~~~~~~~~~~~~~~~~~

A 3-level cascading tool selector for AI agents.

Filter up to 1 000 tools down to the single most relevant one in ~450 ms
at ~$0.0001 per call.

::

    Level 1 -- Embedding : local, ~20 ms,  ~$0
    Level 2 -- Reranker  : local, ~243 ms, ~$0
    Level 3 -- Micro-LLM : API,   ~150 ms, ~$0.0001

Quickstart
----------

.. code-block:: python

    from tool_selector_cascade import CascadeSelector

    selector = CascadeSelector()
    selector.warm_up(all_tools)

    # Sync (Level 1 + 2 only -- no LLM cost)
    candidates, metrics = selector.select(intent, tools, top_k=5)

    # Async (full 3-level cascade)
    result, metrics = await selector.aselect(intent, tools)
    print(result[0].name)
    print(metrics.as_dict())

Convenience function
--------------------

.. code-block:: python

    from tool_selector_cascade import select_tools_for_intent

    # One-liner synchronous filtering (Level 1 + 2, no LLM cost)
    relevant = select_tools_for_intent(intent, tools, top_k=5)
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from tool_selector_cascade.cascade import CascadeSelector
from tool_selector_cascade.config import CascadeConfig
from tool_selector_cascade.metrics import SelectionMetrics

__version__ = "0.1.1"

__all__ = [
    "CascadeSelector",
    "CascadeConfig",
    "SelectionMetrics",
    "select_tools_for_intent",
    "warm_up",
]

# ---------------------------------------------------------------------------
# Convenience function — synchronous one-liner API
# ---------------------------------------------------------------------------

_default_selector: CascadeSelector[Any] | None = None
_default_selector_lock = threading.Lock()


def _get_default_selector() -> CascadeSelector[Any]:
    global _default_selector
    if _default_selector is None:
        with _default_selector_lock:
            if _default_selector is None:
                _default_selector = CascadeSelector()
    return _default_selector


def select_tools_for_intent(
    intent: str,
    tools: Sequence[Any],
    *,
    top_k: int = 5,
    min_threshold: int = 0,
    always_include_prefixes: list[str] | None = None,
) -> list[Any]:
    """Synchronous convenience function for Level 1 + 2 filtering (no LLM cost).

    Uses a module-level :class:`CascadeSelector` singleton so no setup is
    required.  For the full 3-level cascade (including LLM reasoning) use
    :meth:`CascadeSelector.aselect` directly.

    Parameters
    ----------
    intent:
        User task description.
    tools:
        Pool of tools — any duck-typed representation is supported: objects
        with ``.name`` / ``.description`` attributes, plain dicts
        (``{"name": ..., "description": ...}``), or OpenAI function schemas
        (``{"function": {"name": ..., "description": ...}}``).
    top_k:
        Maximum number of tools to return. Default: 5.
    min_threshold:
        Pool size below which filtering is skipped (returns full pool).
    always_include_prefixes:
        Tool name prefixes that are always present in the output regardless
        of their similarity score.

    Returns
    -------
    List
        Filtered and ranked tool list, length <= *top_k*.
    """
    selector = _get_default_selector()
    # Apply per-call overrides on the shared config
    if always_include_prefixes is not None:
        selector.config.always_include_prefixes = always_include_prefixes
    if min_threshold > 0:
        selector.config.min_pool_size = min_threshold
    result, _ = selector.select(intent, tools, top_k=top_k)
    return result


def warm_up(tools: Sequence[Any] | None = None) -> bool:
    """Pre-load Level 1 and Level 2 models for the module-level selector.

    Call this once at application startup (ideally in a background thread) to
    eliminate cold-start latency on the first real query.

    Parameters
    ----------
    tools:
        Optional tool pool to pre-encode.  When provided, the first
        :func:`select_tools_for_intent` call on this exact pool is a cache hit.

    Returns
    -------
    bool
        ``True`` if the Level 1 embedding model loaded successfully.

    Example
    -------
    .. code-block:: python

        import threading
        from tool_selector_cascade import warm_up

        threading.Thread(
            target=warm_up,
            kwargs={"tools": my_tools},
            daemon=True,
            name="cascade-warmup",
        ).start()
    """
    return _get_default_selector().warm_up(tools)
