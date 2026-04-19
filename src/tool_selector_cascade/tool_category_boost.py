"""Category boost configuration for Level-2 reranker scoring.

Infrastructure categories are excluded from semantic competition and assigned
the fixed score configured by ``INFRASTRUCTURE_FIXED_SCORE`` in reranker.py.
"""

from __future__ import annotations

# Categories considered infrastructure/support tools (not task-solving tools).
infrastructure = {
    "infrastructure",
    "infra",
    "system",
    "filesystem",
    "security",
    "context",
    "autonomy",
    "observability",
    "monitoring",
}

# Additive score boosts for task tool categories.
weights = {
    "browser": 0.18,
    "web_automation": 0.18,
    "documents": 0.16,
    "office": 0.16,
    "database": 0.14,
    "network": 0.12,
    "integration": 0.10,
    "messaging": 0.10,
    "social": 0.10,
    "calendar": 0.08,
    "payment": 0.08,
    "media": 0.06,
    "parser": 0.06,
    "cli": 0.04,
}
