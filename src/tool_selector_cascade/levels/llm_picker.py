"""Level 3 -- Micro-LLM final picker.

Sends the top-5 candidates from Level 2 to a fast, cheap LLM and asks it
to reason about which single tool best matches the user intent.

Provider defaults
-----------------
- Anthropic  : ``claude-haiku-4-5``        (~$0.0001 per call, ~150 ms)
- Google     : ``gemini-2.5-flash-lite``    (~$0.0001 per call, ~150 ms)
- OpenAI     : ``gpt-4o-mini``              (~$0.0001 per call, ~200 ms)

Cost model
----------
With 5 candidate (name + description) strings the prompt is typically 300-600
tokens.  At $0.25/M input tokens (Haiku 4.5) this is ~$0.0001 per call.

Graceful degradation
--------------------
If the API key is missing, the provider is unsupported, or the call fails
(network error, timeout, unparseable response), the level transparently returns
the top-1 tool from Level 2 with ``skipped=True`` in the metrics.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from tool_selector_cascade.metrics import LevelMetrics, Timer
from tool_selector_cascade.types import tool_as_text

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a tool selection expert for an AI agent. "
    "Given a user intent and a numbered list of candidate tools, "
    "select the SINGLE most appropriate tool. "
    "Rules: output ONLY the 0-based integer index of the best tool. "
    "No explanation. Just the integer."
)

_USER_TEMPLATE = (
    "User intent: {intent}\n\nCandidate tools:\n{tool_list}\n\n"
    "Which tool index (0-based integer) is MOST relevant? "
    "Respond with ONLY the integer."
)


def _build_tool_list(tools: Sequence[Any]) -> str:
    return "\n".join(f"[{i}] {tool_as_text(t)}" for i, t in enumerate(tools))


def _parse_index(text: str, max_index: int) -> int | None:
    """Extract the first valid 0-based integer from an LLM response."""
    text = text.strip()
    if re.fullmatch(r"\d+", text):
        idx = int(text)
        return idx if 0 <= idx < max_index else None
    # Negative lookbehind prevents extracting digits preceded by '-' (e.g. "-1")
    match = re.search(r"(?<!\-)\b(\d+)\b", text)
    if match:
        idx = int(match.group(1))
        return idx if 0 <= idx < max_index else None
    return None


# ---------------------------------------------------------------------------
# Provider-specific callers
# ---------------------------------------------------------------------------


async def _call_anthropic(
    intent: str,
    tools: Sequence[Any],
    *,
    model: str,
    api_key: str,
    timeout: float,
) -> tuple[int | None, float]:
    """Call Anthropic and return (tool_index, cost_usd)."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model=model,
        max_tokens=16,
        temperature=0.0,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(intent=intent, tool_list=_build_tool_list(tools)),
            }
        ],
        timeout=timeout,
    )
    text = response.content[0].text if response.content else ""
    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0) if usage else 0
    # Haiku 4.5: $0.25/M input, $1.25/M output
    cost = tokens * 0.00000025
    return _parse_index(text, len(tools)), cost


async def _call_openai(
    intent: str,
    tools: Sequence[Any],
    *,
    model: str,
    api_key: str,
    timeout: float,
) -> tuple[int | None, float]:
    """Call OpenAI and return (tool_index, cost_usd)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=16,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(intent=intent, tool_list=_build_tool_list(tools)),
            },
        ],
        timeout=timeout,
    )
    text = response.choices[0].message.content or "" if response.choices else ""
    usage = response.usage
    tokens = (
        getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0) if usage else 0
    )
    # gpt-4o-mini: $0.15/M input
    cost = tokens * 0.00000015
    return _parse_index(text, len(tools)), cost


async def _call_google(
    intent: str,
    tools: Sequence[Any],
    *,
    model: str,
    api_key: str,
    timeout: float,
) -> tuple[int | None, float]:
    """Call Google Generative AI and return (tool_index, cost_usd).

    ``genai.configure()`` sets module-global state in ``google-generativeai``.
    ``_google_lock`` serializes concurrent calls so the configure+call sequence
    is atomic and no concurrent call can overwrite the key before the API
    request completes.
    """
    genai = _get_genai()
    async with _google_lock:
        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel(
            model_name=model,
            system_instruction=_SYSTEM_PROMPT,
        )
        response = await asyncio.wait_for(
            gmodel.generate_content_async(
                _USER_TEMPLATE.format(intent=intent, tool_list=_build_tool_list(tools)),
                generation_config=genai.GenerationConfig(max_output_tokens=16, temperature=0.0),
            ),
            timeout=timeout,
        )
    text = response.text or ""
    # Flat estimate; Gemini Flash Lite is very cheap
    cost = 0.00005
    return _parse_index(text, len(tools)), cost


# ---------------------------------------------------------------------------
# Module-level lock — guards genai.configure() global state (google provider)
# ---------------------------------------------------------------------------

_google_lock: asyncio.Lock = asyncio.Lock()


def _get_genai() -> Any:
    """Import and return the google.generativeai module.

    Extracted as a top-level function so tests can patch it without manipulating
    ``sys.modules``.
    """
    import google.generativeai as genai  # noqa: PLC0415

    return genai


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

# Matches common API key patterns to redact from exception messages
_SECRETS_RE = re.compile(
    r"(sk-[A-Za-z0-9\-_]{20,}|sk-ant-[A-Za-z0-9\-_]{20,}|AIza[A-Za-z0-9\-_]{35,})",
    re.IGNORECASE,
)


def _redact_secrets(text: str) -> str:
    """Replace known API key patterns with [REDACTED] in error messages."""
    return _SECRETS_RE.sub("[REDACTED]", text)


_PROVIDER_CALLERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "google": _call_google,
}
_PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class LLMPickerL3(Generic[T]):
    """Level-3 micro-LLM tool picker.

    Sends the top candidates to a fast, cheap LLM and asks it to reason
    about which single tool best matches the user intent.

    Parameters
    ----------
    provider:
        LLM provider: ``"anthropic"``, ``"openai"``, or ``"google"``.
    model:
        Model identifier within the provider. Default: ``"claude-haiku-4-5"``.
    api_key:
        API key.  If ``None``, reads ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``
        / ``GOOGLE_API_KEY`` from the environment. Default: ``None``.
    timeout:
        HTTP timeout in seconds. Default: 30.0.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.timeout = timeout
        self._api_key = api_key

    def _resolve_api_key(self) -> str | None:
        if self._api_key:
            return self._api_key
        env_var = _PROVIDER_ENV_KEYS.get(self.provider)
        return os.environ.get(env_var) if env_var else None

    async def pick(
        self,
        intent: str,
        tools: Sequence[T],
    ) -> tuple[list[T], LevelMetrics]:
        """Ask the LLM to pick the single best tool from *tools*.

        Falls back to ``tools[0]`` (best reranker score) if the LLM call fails,
        the response is unparseable, or the provider is unsupported.

        Parameters
        ----------
        intent:
            User task description.
        tools:
            Candidate tools from Level 2 (typically top-5).

        Returns
        -------
        tuple[List[T], LevelMetrics]
            List containing the single chosen tool, and level metrics.
        """
        metrics = LevelMetrics(input_count=len(tools))

        if not tools:
            return [], metrics

        if len(tools) == 1:
            metrics.output_count = 1
            metrics.skipped = True
            return list(tools), metrics

        api_key = self._resolve_api_key()
        if not api_key:
            logger.warning(
                "LLMPickerL3: no API key for provider '%s' -- returning top-1 from Level 2",
                self.provider,
            )
            metrics.skipped = True
            metrics.output_count = 1
            return [tools[0]], metrics

        caller = _PROVIDER_CALLERS.get(self.provider)
        if caller is None:
            logger.warning(
                "LLMPickerL3: unsupported provider '%s' -- returning top-1",
                self.provider,
            )
            metrics.skipped = True
            metrics.output_count = 1
            return [tools[0]], metrics

        try:
            with Timer() as timer:
                idx, cost = await caller(
                    intent,
                    tools,
                    model=self.model,
                    api_key=api_key,
                    timeout=self.timeout,
                )

            metrics.latency_ms = timer.elapsed_ms
            metrics.cost_usd = cost

            chosen_idx = idx if idx is not None else 0
            result = [tools[chosen_idx]]
            metrics.output_count = 1

            logger.info(
                "LLMPickerL3: selected tool[%d] '%s' in %.1f ms (cost=$%.5f)",
                chosen_idx,
                getattr(result[0], "name", result[0]),
                metrics.latency_ms,
                cost,
            )
            return result, metrics

        except Exception as exc:
            safe_msg = _redact_secrets(str(exc))
            logger.warning("LLMPickerL3: API call failed '%s' -- fallback to top-1", safe_msg)
            metrics.error = safe_msg
            metrics.output_count = 1
            return [tools[0]], metrics
