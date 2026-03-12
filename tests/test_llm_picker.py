"""Unit tests for the Level 3 micro-LLM picker (LLMPickerL3)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tool_selector_cascade.levels.llm_picker import (
    LLMPickerL3,
    _call_google,
    _parse_index,
    _redact_secrets,
)

_PROVIDER_CALLERS_PATH = "tool_selector_cascade.levels.llm_picker._PROVIDER_CALLERS"
_GET_GENAI_PATH = "tool_selector_cascade.levels.llm_picker._get_genai"


def _make_tool(name: str, description: str = "") -> MagicMock:
    t = MagicMock(spec_set=["name", "description"])
    t.name = name
    t.description = description
    return t


# ---------------------------------------------------------------------------
# _parse_index
# ---------------------------------------------------------------------------


class TestParseIndex:
    def test_exact_integer(self) -> None:
        assert _parse_index("2", 5) == 2

    def test_integer_with_whitespace(self) -> None:
        assert _parse_index("  3  ", 5) == 3

    def test_integer_in_sentence(self) -> None:
        assert _parse_index("The best tool is 1 because it searches.", 5) == 1

    def test_out_of_range_returns_none(self) -> None:
        assert _parse_index("10", 5) is None

    def test_no_digit_returns_none(self) -> None:
        assert _parse_index("no numbers here", 5) is None

    def test_negative_index_returns_none(self) -> None:
        # Negative integers are not matched by \b\d+\b
        assert _parse_index("-1", 5) is None

    def test_zero_is_valid(self) -> None:
        assert _parse_index("0", 3) == 0


# ---------------------------------------------------------------------------
# LLMPickerL3.pick()
# ---------------------------------------------------------------------------


class TestLLMPickerL3Pick:
    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_tools(self) -> None:
        picker = LLMPickerL3()
        result, metrics = await picker.pick("intent", [])
        assert result == []
        assert metrics.input_count == 0

    @pytest.mark.asyncio
    async def test_single_tool_skips_api_call(self) -> None:
        picker = LLMPickerL3()
        tools = [_make_tool("only_tool")]
        result, metrics = await picker.pick("intent", tools)
        assert len(result) == 1
        assert result[0].name == "only_tool"
        assert metrics.skipped is True

    @pytest.mark.asyncio
    async def test_returns_top1_when_no_api_key(self) -> None:
        picker = LLMPickerL3(provider="anthropic", api_key=None)
        tools = [_make_tool(f"t_{i}") for i in range(3)]
        with patch.dict("os.environ", {}, clear=True):
            # No ANTHROPIC_API_KEY in env
            result, metrics = await picker.pick("intent", tools)
        assert len(result) == 1
        assert result[0].name == "t_0"  # top-1 fallback
        assert metrics.skipped is True

    @pytest.mark.asyncio
    async def test_unsupported_provider_returns_top1(self) -> None:
        picker = LLMPickerL3(provider="unknown_provider", api_key="fake")
        tools = [_make_tool(f"t_{i}") for i in range(3)]
        result, metrics = await picker.pick("intent", tools)
        assert result[0].name == "t_0"
        assert metrics.skipped is True

    @pytest.mark.asyncio
    async def test_anthropic_caller_returns_correct_tool(self) -> None:
        picker = LLMPickerL3(provider="anthropic", api_key="sk-fake")
        tools = [_make_tool("gmail_send"), _make_tool("web_search"), _make_tool("calendar")]

        async def _fake_anthropic(intent: Any, tools: Any, **kwargs: Any) -> Any:
            return 1, 0.00001  # picks index 1 = web_search

        with patch.dict(
            "tool_selector_cascade.levels.llm_picker._PROVIDER_CALLERS",
            {"anthropic": _fake_anthropic},
        ):
            result, metrics = await picker.pick("search the web", tools)

        assert result[0].name == "web_search"
        assert metrics.output_count == 1
        assert metrics.cost_usd == pytest.approx(0.00001)

    @pytest.mark.asyncio
    async def test_api_exception_returns_top1_fallback(self) -> None:
        picker = LLMPickerL3(provider="anthropic", api_key="sk-fake")
        tools = [_make_tool(f"t_{i}") for i in range(3)]

        async def _failing_caller(*args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("network unreachable")

        with patch.dict(
            "tool_selector_cascade.levels.llm_picker._PROVIDER_CALLERS",
            {"anthropic": _failing_caller},
        ):
            result, metrics = await picker.pick("intent", tools)

        assert result[0].name == "t_0"
        assert metrics.error is not None

    @pytest.mark.asyncio
    async def test_unparseable_response_falls_back_to_index_0(self) -> None:
        picker = LLMPickerL3(provider="anthropic", api_key="sk-fake")
        tools = [_make_tool(f"t_{i}") for i in range(3)]

        async def _bad_response(*args: Any, **kwargs: Any) -> Any:
            return None, 0.00001  # _parse_index returned None

        with patch.dict(
            "tool_selector_cascade.levels.llm_picker._PROVIDER_CALLERS",
            {"anthropic": _bad_response},
        ):
            result, metrics = await picker.pick("intent", tools)

        assert result[0].name == "t_0"  # fallback to index 0


# ---------------------------------------------------------------------------
# _redact_secrets (SEC-03 helper)
# ---------------------------------------------------------------------------


class TestRedactSecrets:
    """Unit tests for the _redact_secrets() sanitization helper."""

    def test_openai_key_is_redacted(self) -> None:
        """sk- prefixed keys are replaced with [REDACTED]."""
        msg = "Authentication failed: sk-abcdefghij1234567890abcd"
        result = _redact_secrets(msg)
        assert "sk-abcdefghij1234567890abcd" not in result
        assert "[REDACTED]" in result

    def test_anthropic_key_is_redacted(self) -> None:
        """sk-ant- prefixed keys are replaced with [REDACTED]."""
        msg = "Invalid API key: sk-ant-api03-FakeXxxYyyZzz1234567890ABCDEF"
        result = _redact_secrets(msg)
        assert "sk-ant-api03-FakeXxxYyyZzz1234567890ABCDEF" not in result
        assert "[REDACTED]" in result

    def test_google_api_key_is_redacted(self) -> None:
        """AIza-prefixed keys (Google API keys) are replaced with [REDACTED]."""
        # Real Google API key format: AIza + 35 alphanumeric chars
        msg = "Google auth error: AIzaSyFakeGoogleKey1234567890abcdefghij"
        result = _redact_secrets(msg)
        assert "AIzaSyFakeGoogleKey1234567890abcdefghij" not in result
        assert "[REDACTED]" in result

    def test_non_secret_text_is_unchanged(self) -> None:
        """Regular error messages without API keys are not modified."""
        msg = "ConnectionError: network unreachable at 192.168.1.1:443"
        assert _redact_secrets(msg) == msg

    def test_key_embedded_in_sentence_is_fully_redacted(self) -> None:
        """A key embedded mid-sentence is fully replaced, surrounding text preserved."""
        msg = "Error: Authorization header Bearer sk-TestKey12345678901234 is invalid"
        result = _redact_secrets(msg)
        assert "sk-TestKey12345678901234" not in result
        assert "Authorization header Bearer" in result
        assert "[REDACTED]" in result

    def test_multiple_keys_in_one_message_all_redacted(self) -> None:
        """Multiple API key patterns in a single message are all replaced."""
        msg = (
            "OpenAI key sk-openai12345678901234567890, "
            "Anthropic key sk-ant-FakeAnthropicKey1234567890ABCDEF are both invalid"
        )
        result = _redact_secrets(msg)
        assert "sk-openai12345678901234567890" not in result
        assert "sk-ant-FakeAnthropicKey1234567890ABCDEF" not in result
        assert result.count("[REDACTED]") == 2


# ---------------------------------------------------------------------------
# _call_google security fixes (SEC-01 lock, SEC-02 timeout)
# ---------------------------------------------------------------------------


class TestCallGoogleSecFixes:
    """Tests verifying SEC-01 (_google_lock) and SEC-02 (timeout) fixes in _call_google."""

    @pytest.mark.asyncio
    async def test_call_google_successful_invocation_returns_correct_index(self) -> None:
        """_call_google returns (index, cost) correctly after security fixes (regression)."""

        class _FakeModel:
            async def generate_content_async(self, *args: Any, **kwargs: Any) -> Any:
                r = MagicMock()
                r.text = "2"
                return r

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = _FakeModel()
        mock_genai.GenerationConfig.return_value = {}

        tools = [_make_tool(f"t_{i}") for i in range(5)]

        with patch(_GET_GENAI_PATH, return_value=mock_genai):
            idx, cost = await _call_google(
                "intent", tools, model="gemini-flash-lite", api_key="test-key", timeout=5.0
            )

        assert idx == 2
        assert cost == pytest.approx(0.00005)
        mock_genai.configure.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_call_google_timeout_raises_asyncio_timeout_error(self) -> None:
        """SEC-02: asyncio.wait_for enforces the timeout parameter in _call_google."""

        class _HangingModel:
            async def generate_content_async(self, *args: Any, **kwargs: Any) -> Any:
                await asyncio.sleep(10)  # Simulate an unresponsive API
                return MagicMock()

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = _HangingModel()
        mock_genai.GenerationConfig.return_value = {}

        tools = [_make_tool("t_0"), _make_tool("t_1")]

        with patch(_GET_GENAI_PATH, return_value=mock_genai):
            with pytest.raises(asyncio.TimeoutError):
                await _call_google(
                    "intent", tools, model="gemini-flash", api_key="KEY", timeout=0.01
                )

    @pytest.mark.asyncio
    async def test_google_lock_serializes_configure_and_generate(self) -> None:
        """SEC-01: _google_lock ensures configure() and generate_content_async() are atomic.

        Two concurrent calls with different API keys must never interleave:
        configure(KEY_A) followed by generate_A must fully complete before
        configure(KEY_B) runs.
        """
        events: list = []
        call_count = [0]

        def _fake_configure(api_key: str) -> None:
            call_count[0] += 1
            events.append(f"configure_{call_count[0]}")

        def _fake_model_factory(*args: Any, **kwargs: Any) -> Any:
            if call_count[0] == 1:

                class _SlowModel:
                    async def generate_content_async(self, *a: Any, **kw: Any) -> Any:
                        events.append("gen_1_start")
                        await asyncio.sleep(0.02)  # Yield so Task B tries to interleave
                        events.append("gen_1_end")
                        r = MagicMock()
                        r.text = "0"
                        return r

                return _SlowModel()
            else:

                class _FastModel:
                    async def generate_content_async(self, *a: Any, **kw: Any) -> Any:
                        events.append("gen_2_start")
                        r = MagicMock()
                        r.text = "0"
                        return r

                return _FastModel()

        mock_genai = MagicMock()
        mock_genai.configure.side_effect = _fake_configure
        mock_genai.GenerativeModel.side_effect = _fake_model_factory
        mock_genai.GenerationConfig.return_value = {}

        tools = [_make_tool("t_0"), _make_tool("t_1")]

        with patch(_GET_GENAI_PATH, return_value=mock_genai):
            await asyncio.gather(
                asyncio.create_task(
                    _call_google("intent_a", tools, model="m", api_key="KEY_A", timeout=5.0)
                ),
                asyncio.create_task(
                    _call_google("intent_b", tools, model="m", api_key="KEY_B", timeout=5.0)
                ),
            )

        # Lock guarantees: configure_1 → gen_1_start → gen_1_end → configure_2 → gen_2_start
        assert events.index("configure_1") < events.index("gen_1_start")
        assert events.index("gen_1_end") < events.index("configure_2"), (
            "configure_2 must not run while gen_1 is still executing (lock violation)"
        )
        assert events.index("configure_2") < events.index("gen_2_start")

    @pytest.mark.asyncio
    async def test_google_lock_released_on_timeout(self) -> None:
        """SEC-01 + SEC-02: _google_lock is released when a timeout occurs.

        If the lock were kept after a timeout, the next call would deadlock.
        """
        call_count = [0]

        def _model_factory(*args: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:

                class _HangModel:
                    async def generate_content_async(self, *a: Any, **kw: Any) -> Any:
                        await asyncio.sleep(10)
                        return MagicMock()

                return _HangModel()
            else:

                class _FastModel:
                    async def generate_content_async(self, *a: Any, **kw: Any) -> Any:
                        r = MagicMock()
                        r.text = "0"
                        return r

                return _FastModel()

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.side_effect = _model_factory
        mock_genai.GenerationConfig.return_value = {}

        tools = [_make_tool("t_0"), _make_tool("t_1")]

        with patch(_GET_GENAI_PATH, return_value=mock_genai):
            # First call times out — lock must be released afterwards
            with pytest.raises(asyncio.TimeoutError):
                await _call_google("intent", tools, model="m", api_key="K1", timeout=0.01)

            # Second call must NOT deadlock (proves the lock was released)
            idx, cost = await _call_google("intent", tools, model="m", api_key="K2", timeout=5.0)

        assert idx == 0  # fast model returns "0"
        assert cost == pytest.approx(0.00005)


# ---------------------------------------------------------------------------
# pick() security integration (SEC-01 + SEC-02 + SEC-03)
# ---------------------------------------------------------------------------


class TestPickSecurityFixes:
    """Integration tests for all security fixes applied to LLMPickerL3.pick()."""

    @pytest.mark.asyncio
    async def test_pick_google_provider_routes_to_call_google(self) -> None:
        """pick() with provider='google' dispatches to _call_google."""
        picker = LLMPickerL3(
            provider="google",
            api_key="AIzaFakeGoogleKey12345678901234567890",
        )
        tools = [_make_tool(f"t_{i}") for i in range(3)]

        async def _fake_google(*args: Any, **kwargs: Any) -> Any:
            return 2, 0.00005

        with patch.dict(
            "tool_selector_cascade.levels.llm_picker._PROVIDER_CALLERS",
            {"google": _fake_google},
        ):
            result, metrics = await picker.pick("intent", tools)

        assert result[0].name == "t_2"
        assert metrics.cost_usd == pytest.approx(0.00005)
        assert metrics.output_count == 1

    @pytest.mark.asyncio
    async def test_pick_google_timeout_falls_back_to_top1(self) -> None:
        """SEC-02: pick() falls back gracefully when _call_google raises TimeoutError."""
        picker = LLMPickerL3(
            provider="google",
            api_key="AIzaSyFakeGoogleKey1234567890abcdefghij",
        )
        tools = [_make_tool(f"t_{i}") for i in range(3)]

        async def _timeout_google(*args: Any, **kwargs: Any) -> Any:
            raise TimeoutError()

        with patch.dict(
            "tool_selector_cascade.levels.llm_picker._PROVIDER_CALLERS",
            {"google": _timeout_google},
        ):
            result, metrics = await picker.pick("intent", tools)

        assert result[0].name == "t_0"  # graceful fallback to Level 2 top-1
        assert metrics.error is not None  # error recorded (TimeoutError str may be empty)

    @pytest.mark.asyncio
    async def test_pick_openai_key_in_exception_is_redacted(self) -> None:
        """SEC-03: API key leaked via SDK exception message is redacted in metrics.error."""
        secret_key = "sk-SecretOpenAIKey1234567890abcdef"
        picker = LLMPickerL3(provider="anthropic", api_key=secret_key)
        tools = [_make_tool(f"t_{i}") for i in range(3)]

        async def _leaky_caller(*args: Any, **kwargs: Any) -> Any:
            raise ValueError(
                f"Authentication failed: invalid key {secret_key}, please check your account."
            )

        with patch.dict(
            "tool_selector_cascade.levels.llm_picker._PROVIDER_CALLERS",
            {"anthropic": _leaky_caller},
        ):
            result, metrics = await picker.pick("intent", tools)

        assert result[0].name == "t_0"
        assert secret_key not in (metrics.error or ""), (
            "Raw API key must not appear in metrics.error"
        )
        assert "[REDACTED]" in (metrics.error or "")

    @pytest.mark.asyncio
    async def test_pick_google_key_in_exception_is_redacted(self) -> None:
        """SEC-03: Google AIza key in exception message is redacted in metrics.error."""
        secret_key = "AIzaSyFakeGoogleKey1234567890abcdefghij"
        picker = LLMPickerL3(provider="google", api_key=secret_key)
        tools = [_make_tool(f"t_{i}") for i in range(3)]

        async def _leaky_caller(*args: Any, **kwargs: Any) -> Any:
            raise PermissionError(f"Forbidden: key {secret_key} lacks required scope")

        with patch.dict(
            "tool_selector_cascade.levels.llm_picker._PROVIDER_CALLERS",
            {"google": _leaky_caller},
        ):
            result, metrics = await picker.pick("intent", tools)

        assert result[0].name == "t_0"
        assert secret_key not in (metrics.error or "")
        assert "[REDACTED]" in (metrics.error or "")

    @pytest.mark.asyncio
    async def test_pick_regular_exception_message_preserved(self) -> None:
        """SEC-03 (regression): non-secret error messages are not modified."""
        picker = LLMPickerL3(provider="anthropic", api_key="sk-fake12345678901234")
        tools = [_make_tool(f"t_{i}") for i in range(3)]
        expected_msg = "ConnectionError: network unreachable at 10.0.0.1:443"

        async def _network_error(*args: Any, **kwargs: Any) -> Any:
            raise ConnectionError(expected_msg)

        with patch.dict(
            "tool_selector_cascade.levels.llm_picker._PROVIDER_CALLERS",
            {"anthropic": _network_error},
        ):
            result, metrics = await picker.pick("intent", tools)

        assert result[0].name == "t_0"
        assert metrics.error == expected_msg
