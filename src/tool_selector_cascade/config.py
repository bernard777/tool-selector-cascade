"""Configuration dataclass for :class:`CascadeSelector`.

All fields expose sensible defaults matching the recommended model sizes
described in the package README.  Override only what you need::

    config = CascadeConfig(
        llm_provider="google",
        llm_model="gemini-2.5-flash-lite",
        reranker_top_k=3,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CascadeConfig:
    """Full configuration for the 3-level cascade selector.

    All fields have sensible defaults; override only what you need.

    Parameters
    ----------
    embedder_model:
        HuggingFace bi-encoder model (Level 1).
        Default: ``intfloat/multilingual-e5-base`` (~270 MB, ~20 ms).
    embedder_top_k:
        Maximum candidates forwarded from Level 1 to Level 2. Default: 20.
    embedder_min_pool:
        Pool size below which Level 1 is skipped. Default: 20.
    reranker_model:
        HuggingFace cross-encoder model (Level 2).
        Default: ``Alibaba-NLP/gte-reranker-modernbert-base`` (~300 MB, ~243 ms).
    reranker_top_k:
        Maximum candidates forwarded from Level 2 to Level 3. Default: 5.
    reranker_enabled:
        Set to ``False`` to bypass Level 2 entirely. Default: ``True``.
    llm_provider:
        Level 3 provider: ``"anthropic"`` | ``"google"`` | ``"openai"``.
    llm_model:
        Model identifier within the chosen provider. Default: ``"claude-haiku-4-5"``.
    llm_api_key:
        API key; reads provider env var if ``None``. Default: ``None``.
    llm_enabled:
        Set to ``False`` to skip Level 3 and return Level 2 result. Default: ``True``.
    llm_timeout:
        HTTP timeout in seconds for Level 3 calls. Default: 30.0.
    always_include_prefixes:
        Tool name prefixes guaranteed a slot in the output, regardless of score.
    min_pool_size:
        Pool size below which *all* filtering is skipped. Default: 5.
    """

    # Level 1 -- Embedding
    embedder_model: str = "intfloat/multilingual-e5-base"
    embedder_top_k: int = 20
    embedder_min_pool: int = 20

    # Level 2 -- Reranker
    # NOTE: Alibaba-NLP/gte-reranker-modernbert-base requires transformers>=4.49
    # (ModernBERT architecture).  The project pins transformers<4.47 for OmniParser
    # compatibility, so we use a BERT-based cross-encoder that works on 4.46.x.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5
    reranker_enabled: bool = True

    # Level 3 -- Micro-LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-4-5"
    llm_api_key: str | None = None
    llm_enabled: bool = True
    llm_timeout: float = 30.0

    # General
    always_include_prefixes: list[str] = field(default_factory=list)
    min_pool_size: int = 5
