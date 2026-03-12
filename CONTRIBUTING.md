# Contributing to tool-selector-cascade

Thank you for taking the time to contribute! This guide covers everything you
need to know to get your PR merged smoothly.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Branch Strategy](#branch-strategy)
4. [Coding Standards](#coding-standards)
5. [Tests](#tests)
6. [Documentation](#documentation)
7. [Changelog](#changelog)
8. [Pull Request Process](#pull-request-process)
9. [Release Process](#release-process)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you agree to uphold these standards.

---

## Getting Started

```bash
# 1. Fork the repository and clone your fork
git clone https://github.com/<your-username>/tool-selector-cascade.git
cd tool-selector-cascade

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\Activate.ps1       # Windows PowerShell

# 3. Install in editable mode with all development dependencies
pip install -e ".[dev,all]"

# 4. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 5. Run the test suite to confirm everything works
pytest tests/ -v
```

---

## Branch Strategy

```
master  ← protected, production-ready releases only
  └── dev  ← integration branch, all features merge here first
        └── feature/<short-name>    your work
        └── fix/<issue-id>-description
        └── docs/<topic>
        └── refactor/<scope>
```

- **Open PRs against `dev`**, not `master`.
- `master` receives PRs only from `dev` as release commits.
- Name your branch `feature/<short-name>`, `fix/<issue>-<desc>`, or `docs/<topic>`.

---

## Coding Standards

### Python
- **Python 3.11+** only.
- **Type hints everywhere** — `mypy --strict` must pass with zero errors.
- **Async/await** for all I/O operations; never block the event loop.
- **Structured logging** — use positional args: `logger.warning("msg {}", exc)`,
  never f-strings in log calls.
- **PEP 8** enforced by `ruff` and `black` (line length: 100).
- **Import ordering** enforced by `isort` (profile: black).

### Running linters locally

```bash
ruff check .
black --check .
isort --check-only .
mypy src/ --strict
```

Or simply run pre-commit which handles all of the above:

```bash
pre-commit run --all-files
```

### Security
- Never hardcode API keys, tokens, or credentials. Use environment variables.
- API key patterns in exception messages must be redacted (use `_redact_secrets()`).
- Protect shared async state with `asyncio.Lock`.
- Always enforce timeouts on external HTTP calls.

---

## Tests

All new code **must** include tests. The test suite uses `pytest` and `pytest-asyncio`.

```bash
# Run all tests
pytest tests/ -v

# Run with coverage (must stay ≥ 80%)
pytest tests/ --cov=src/tool_selector_cascade --cov-report=term-missing --cov-fail-under=80

# Run a specific file
pytest tests/test_llm_picker.py -v
```

### Test conventions
- Use `@pytest.mark.asyncio` for async tests.
- Mock external calls with `patch.dict(module._PROVIDER_CALLERS, {...})`.
- Use `MagicMock(spec_set=["name", "description"])` for tool mocks.
- Each test class maps to one unit/component; one test = one assertion focus.
- Never share mutable state between tests.

---

## Documentation

- **Docstrings** for all public functions and classes (Google style).
- If you change the public API, update `README.md` accordingly.
- Update `examples/basic_usage.py` if usage patterns change.

---

## Changelog

Every PR **must** update `CHANGELOG.md`:

1. Add an entry under `## [Unreleased]` at the top of the file.
2. Use the correct category: `Added`, `Changed`, `Fixed`, `Security`, `Deprecated`, `Removed`.
3. Write in present-tense imperative: *"Add support for…"*, *"Fix race condition in…"*.

The AI PR reviewer will block the PR if `CHANGELOG.md` is not updated.

---

## Pull Request Process

1. Ensure all CI checks pass (lint, type check, tests, build, security).
2. Fill in the PR template completely.
3. The AI reviewer (`ai-pr-review.yml`) will analyze your PR automatically.
4. Address all `REQUEST_CHANGES` findings before requesting a human review.
5. A maintainer will merge once the AI reviewer approves and at least one
   human reviewer has signed off.

---

## Release Process

Releases are done by maintainers only:

1. Merge `dev` → `master` via a release PR.
2. Update `pyproject.toml` version and add a `CHANGELOG.md` release section.
3. Tag the commit: `git tag v0.2.0 && git push origin v0.2.0`.
4. The `release.yml` workflow builds and publishes to PyPI automatically.

---

## Questions?

Open a [Discussion](https://github.com/bernard777/tool-selector-cascade/discussions)
or ping a maintainer in the PR comments.
