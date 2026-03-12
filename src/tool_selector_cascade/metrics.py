"""Metrics collection for the cascade selector pipeline.

Tracks latency and estimated cost for each cascade level so callers can
monitor performance and set up alerts without adding external dependencies.

Example
-------
.. code-block:: python

    result, metrics = await selector.aselect(intent, tools)
    print(metrics.as_dict())
    # {
    #   "total_latency_ms": 423.1,
    #   "total_cost_usd": 0.000085,
    #   "input_pool_size": 235,
    #   "final_count": 1,
    #   "levels": {
    #     "level1_embedding": {"latency_ms": 18.4, "input": 235, "output": 20, ...},
    #     "level2_reranker":  {"latency_ms": 251.2, "input": 20, "output": 5, ...},
    #     "level3_llm":       {"latency_ms": 153.5, "input": 5, "output": 1, ...},
    #   },
    # }
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LevelMetrics:
    """Timing and cost metrics for a single cascade level.

    Attributes
    ----------
    latency_ms:
        Wall-clock time spent in this level, in milliseconds.
    input_count:
        Number of tools entering this level.
    output_count:
        Number of tools exiting this level.
    cost_usd:
        Estimated API cost in USD (0.0 for local inference).
    skipped:
        ``True`` when this level was bypassed (e.g. pool too small, model unavailable).
    error:
        Error message if the level raised an exception but recovered gracefully.
    """

    latency_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    cost_usd: float = 0.0
    skipped: bool = False
    error: str | None = None


@dataclass
class SelectionMetrics:
    """Aggregate metrics for a complete cascade run.

    Attributes
    ----------
    total_latency_ms:
        Sum of all level latencies.
    total_cost_usd:
        Sum of all level costs (Level 3 API calls are the primary contributor).
    input_pool_size:
        Size of the tool pool at the start of the cascade.
    final_count:
        Number of tools returned by the last active level.
    levels:
        Per-level breakdown keyed by level name
        (``"level1_embedding"``, ``"level2_reranker"``, ``"level3_llm"``).
    """

    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    input_pool_size: int = 0
    final_count: int = 0
    levels: dict[str, LevelMetrics] = field(default_factory=dict)

    def add_level(self, name: str, metrics: LevelMetrics) -> None:
        """Register metrics for one level and accumulate totals."""
        self.levels[name] = metrics
        self.total_latency_ms += metrics.latency_ms
        self.total_cost_usd += metrics.cost_usd

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of these metrics."""
        return {
            "total_latency_ms": round(self.total_latency_ms, 1),
            "total_cost_usd": round(self.total_cost_usd, 8),
            "input_pool_size": self.input_pool_size,
            "final_count": self.final_count,
            "levels": {
                name: {
                    "latency_ms": round(m.latency_ms, 1),
                    "input": m.input_count,
                    "output": m.output_count,
                    "cost_usd": round(m.cost_usd, 8),
                    "skipped": m.skipped,
                    **({"error": m.error} if m.error else {}),
                }
                for name, m in self.levels.items()
            },
        }


class Timer:
    """Context manager that measures elapsed wall-clock time in milliseconds.

    Example
    -------
    .. code-block:: python

        with Timer() as t:
            do_work()
        print(t.elapsed_ms)  # e.g. 18.4
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
