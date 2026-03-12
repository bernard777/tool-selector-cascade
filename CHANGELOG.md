# Changelog

All notable changes to `tool-selector-cascade` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.1] - 2026-03-12

### Fixed

- **SEC-01 — asyncio race condition in `_call_google`**: replaced global
  `genai.configure()` with an `asyncio.Lock` (`_google_lock`) so that
  concurrent coroutines cannot interleave `configure()` / `generate_content_async()`
  with different API keys.

- **SEC-02 — timeout enforcement in `_call_google`**: the `timeout` parameter
  was accepted but never enforced; wrapped `generate_content_async()` in
  `asyncio.wait_for(coro, timeout=timeout)` so slow Google responses are
  cancelled and the cascade falls back to Level 2 in bounded time.

- **SEC-03 — API key leakage in exception messages**: added `_redact_secrets()`
  using a compiled regex (`_SECRETS_RE`) that scrubs OpenAI (`sk-…`),
  Anthropic (`sk-ant-…`) and Google (`AIza…`) key patterns before any
  exception message is written to logs or returned through metrics.

- **`EmbedderL1` fallback paths respect `forced_indices`**: the `model is None`
  branch and the exception-catch branch now both include forced tools in the
  returned slice (previously they returned `tools[:top_k]` ignoring the
  forced list, causing `always_include_prefixes` to be silently dropped when
  the embedding model is unavailable).

- **`EmbedderL1` exception safety for `_get_model`**: moved the `_get_model()`
  call inside the outer `try/except` so that exceptions raised during model
  loading (e.g. in tests with `side_effect=RuntimeError`) are caught and
  handled as graceful fallbacks instead of propagating uncaught to the caller.

### Changed

- Extracted `_get_genai()` helper in `llm_picker.py` to make the
  `google-generativeai` import patchable in unit tests without manipulating
  `sys.modules`.

### Tests

- Added 23 new tests across three classes:
  - `TestRedactSecrets` (6 tests) — covers OpenAI, Anthropic and Google key
    patterns, embedded keys, multi-key messages and plain text passthrough.
  - `TestCallGoogleSecFixes` (4 tests) — verifies `asyncio.Lock` serialisation,
    lock release on timeout, `asyncio.TimeoutError` propagation, and correct
    index return on success.
  - `TestPickSecurityFixes` (5 tests) — integration tests verifying that
    `pick()` routes to `_call_google`, falls back on timeout, and redacts
    secret-bearing exception messages (OpenAI, Google, plain errors).
- Fixed 3 pre-existing tests that patched `_call_anthropic` via `side_effect`
  (ineffective because `_PROVIDER_CALLERS` holds direct function references);
  switched to `patch.dict(_PROVIDER_CALLERS, ...)`.
- Fixed `test_forced_indices_always_in_result`: corrected mock `side_effect`
  shape (`intent_emb` shape `(1, 8)` instead of `(8,)`) so that
  `encode()[0]` returns a proper vector, not a 0-d scalar.

**Total test coverage: 58 tests, 0 failures.**

---

## [0.1.0] - 2026-03-12

### Added

- **Level 1 — Embedding filter** (`EmbedderL1`)
  - Bi-encoder cosine similarity using `intfloat/multilingual-e5-base`
  - Per-pool embedding cache keyed by MD5 hash (near-instant repeated calls)
  - `warm_up()` method for pre-loading at application startup
  - Graceful fallback: returns `tools[:top_k]` when model is unavailable

- **Level 2 — Cross-encoder reranker** (`RerankerL2`)
  - Cross-attention re-scoring using `Alibaba-NLP/gte-reranker-modernbert-base`
  - Configurable `top_k` forwarded to Level 3
  - Graceful fallback: returns Level 1 result slice when model is unavailable

- **Level 3 — Micro-LLM picker** (`LLMPickerL3`)
  - Multi-provider support: Anthropic (Haiku), Google (Flash Lite), OpenAI (mini)
  - Structured prompt with 0-based index output format
  - Graceful fallback: returns top Level 2 candidate when API is unavailable

- **CascadeSelector** — main orchestrator
  - Sync API (`select()`) for Level 1 + 2 without LLM cost
  - Async API (`aselect()`) for the full 3-level cascade
  - `always_include_prefixes` guarantee (e.g. `["web_search"]`)
  - Per-call `top_k` override

- **`select_tools_for_intent()`** — synchronous one-liner convenience function
  (Level 1 + 2, no LLM cost) with `top_k`, `min_threshold`, and
  `always_include_prefixes` parameters

- **`warm_up(tools=None)`** — module-level pre-load helper; call once at
  startup (typically in a daemon thread) to eliminate cold-start latency on
  the first query

- **SelectionMetrics** — JSON-serialisable cascade profiling
  (latency, cost, input/output counts per level)

- Full test suite: 35+ unit + integration tests (pytest + pytest-asyncio)

- MIT License

