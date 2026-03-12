#!/usr/bin/env python3
"""AI-powered Pull Request reviewer for tool-selector-cascade.

Called by the GitHub Actions workflow ``ai-pr-review.yml``.

Responsibilities
----------------
1. Fetch PR metadata and unified diff from GitHub REST API.
2. Send the diff to GitHub Models (gpt-4o via OpenAI-compatible endpoint)
   with a structured expert prompt.
3. Receive a JSON-structured review across 5 dimensions:
   - Architecture & design patterns
   - Security (OWASP Top 10)
   - Code quality (Python best practices, types, async)
   - Tests (coverage, edge cases, mocking patterns)
   - Documentation & CHANGELOG completeness
4. Post the review on the PR as an official GitHub review
   (action: APPROVE or REQUEST_CHANGES).

Required environment variables
--------------------------------
GITHUB_TOKEN   Automatically injected by GitHub Actions — used for BOTH
               GitHub API calls AND GitHub Models inference.
               NO additional secret is required.
PR_NUMBER      Pull request number (set by workflow).
REPO           Repository slug, e.g. "bernard777/tool-selector-cascade".

Usage
-----
    python .github/scripts/ai_pr_reviewer.py --pr 42 --repo bernard777/tool-selector-cascade
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Any

from openai import OpenAI
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GitHub Models endpoint — OpenAI-compatible, authenticated via GITHUB_TOKEN.
# No separate secret needed: GITHUB_TOKEN is auto-injected in every workflow run.
GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
MODEL = "gpt-4o"          # Change to "gpt-4o-mini" for higher rate limits
MAX_DIFF_CHARS = 60_000    # Truncate very large diffs to stay within context
GITHUB_API = "https://api.github.com"

# ---------------------------------------------------------------------------
# System prompt — Claude acts as a multi-role senior reviewer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are a senior code reviewer for the open-source Python library **tool-selector-cascade** — a
production-grade 3-level cascading tool selector for AI agents (embedding → reranker → micro-LLM).

Your task: perform an exhaustive code review of the provided Pull Request diff across **5 dimensions**.
Return your review as a JSON object matching the schema at the end of this prompt.

## Dimension 1 — ARCHITECTURE
Check for:
- Separation of concerns (routes / services / core engines are separate)
- Module cohesion and coupling (avoid circular imports)
- Correct use of dependency injection and factory patterns
- Adherence to existing conventions (CascadeSelector, levels/, config.py patterns)
- No backward-incompatible API changes without a version bump
- No spaghetti code or god-classes

## Dimension 2 — SECURITY (OWASP Top 10 focus)
Check for:
- A01 — Broken Access Control (no privilege escalation)
- A02 — Cryptographic failures (no weak hashing, no plaintext secrets)
- A03 — Injection (no f-string/format in exec/eval/subprocess)
- A06 — Vulnerable components (new deps must be pinned and audited)
- A07 — Auth failures (API keys handled via env vars, never hardcoded)
- Secret leakage in logs or exception messages (use _redact_secrets() pattern)
- asyncio race conditions (shared global state must be protected by locks)
- Timeout enforcement on all external HTTP/API calls

## Dimension 3 — CODE QUALITY
Check for:
- Full type hints throughout (mypy strict mode must pass)
- Async/await correctness (no blocking I/O in async context)
- Proper error handling with graceful degradation
- Structured logging (no f-strings in log arguments — positional args only)
- PEP 8 compliance (enforced by ruff + black + isort)
- Function/class size (single responsibility principle)
- No dead code, unused imports, TODO comments without an issue reference

## Dimension 4 — TESTS
Check for:
- Tests provided for all new code paths
- Async fixtures use @pytest.mark.asyncio correctly
- Mocking patterns consistent with existing test suite (patch.dict for callers)
- Edge cases covered (empty input, model unavailable, API timeouts, exceptions)
- Test isolation (no shared mutable state between tests)
- Coverage regression (must not drop below 80%)

## Dimension 5 — DOCUMENTATION
Check for:
- Docstrings on all public functions / classes (Google-style)
- CHANGELOG.md updated with the correct version section
- README.md updated if public API or config changed
- Examples updated if behavior changed

## Rating scale
- PASS     — No issues found in this dimension
- WARNING  — Minor issues that should be addressed but are not blocking
- FAIL     — Blocking issues: correctness, security, or quality problems

## Final verdict
- APPROVE          — All dimensions are PASS or WARNING only, no FAIL
- REQUEST_CHANGES  — At least one dimension is FAIL

## JSON schema (return ONLY valid JSON, no markdown fences)
{
  "dimensions": {
    "architecture": {
      "rating": "PASS|WARNING|FAIL",
      "summary": "<one sentence>",
      "findings": [
        {"file": "<path>", "line": <int or null>, "severity": "info|warning|error", "message": "<description>"}
      ]
    },
    "security": { "rating": "...", "summary": "...", "findings": [...] },
    "code_quality": { "rating": "...", "summary": "...", "findings": [...] },
    "tests": { "rating": "...", "summary": "...", "findings": [...] },
    "documentation": { "rating": "...", "summary": "...", "findings": [...] }
  },
  "verdict": "APPROVE|REQUEST_CHANGES",
  "overall_summary": "<2-3 sentences summarising the PR and the review outcome>"
}
""").strip()


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _gh_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def fetch_pr_metadata(repo: str, pr_number: int, token: str) -> dict[str, Any]:
    """Return PR title, body and base/head SHAs."""
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
    resp = requests.get(url, headers=_gh_headers(token), timeout=30)
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def fetch_pr_diff(repo: str, pr_number: int, token: str) -> str:
    """Return the unified diff of the PR (text/plain)."""
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
    headers = _gh_headers(token)
    headers["Accept"] = "application/vnd.github.v3.diff"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_pr_files(repo: str, pr_number: int, token: str) -> list[dict[str, Any]]:
    """Return list of changed files with stats."""
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files"
    resp = requests.get(url, headers=_gh_headers(token), timeout=30)
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def post_pr_review(
    repo: str,
    pr_number: int,
    token: str,
    body: str,
    action: str,  # "APPROVE" | "REQUEST_CHANGES" | "COMMENT"
) -> None:
    """Post an official PR review via GitHub API."""
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews"
    payload = {"body": body, "event": action}
    resp = requests.post(url, headers=_gh_headers(token), json=payload, timeout=30)
    resp.raise_for_status()
    print(f"[AI Review] Posted review: {action}")


# ---------------------------------------------------------------------------
# AI review logic
# ---------------------------------------------------------------------------

def build_user_message(
    pr_meta: dict[str, Any],
    diff: str,
    files: list[dict[str, Any]],
) -> str:
    """Build the full user message for Claude with PR context + diff."""
    title = pr_meta.get("title", "(no title)")
    body = pr_meta.get("body") or "(no description)"
    base_branch = pr_meta.get("base", {}).get("ref", "?")
    head_branch = pr_meta.get("head", {}).get("ref", "?")
    additions = pr_meta.get("additions", 0)
    deletions = pr_meta.get("deletions", 0)
    changed_files = pr_meta.get("changed_files", 0)

    file_summary = "\n".join(
        f"  - {f['filename']} (+{f['additions']} -{f['deletions']}, status: {f['status']})"
        for f in files[:50]  # cap at 50 files
    )

    # Truncate diff if too large
    truncated = ""
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS]
        truncated = f"\n\n[DIFF TRUNCATED at {MAX_DIFF_CHARS} chars]"

    return textwrap.dedent(f"""
        ## Pull Request: {title}

        **Branch:** `{head_branch}` → `{base_branch}`
        **Stats:** +{additions} -{deletions} lines across {changed_files} file(s)

        ### Description
        {body}

        ### Changed Files
        {file_summary}

        ### Unified Diff
        ```diff
        {diff}
        ```
        {truncated}

        ---
        Please review the above diff according to your instructions and return the JSON review.
    """).strip()


def call_github_model(user_message: str, github_token: str) -> dict[str, Any]:
    """Call GitHub Models (OpenAI-compatible) and parse the JSON review response.

    Authentication is done with the auto-injected GITHUB_TOKEN — no additional
    API key or GitHub secret is required.
    """
    client = OpenAI(
        base_url=GITHUB_MODELS_ENDPOINT,
        api_key=github_token,
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content or ""

    # Strip any accidental markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    return json.loads(raw)  # type: ignore[no-any-return]


def format_github_review(review: dict[str, Any]) -> str:
    """Convert the structured review JSON into a readable GitHub review comment."""
    dims = review.get("dimensions", {})
    verdict = review.get("verdict", "COMMENT")
    overall = review.get("overall_summary", "")

    rating_emoji = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}

    lines: list[str] = []
    lines.append("## 🤖 AI Code Review — `tool-selector-cascade`")
    lines.append("")
    lines.append(f"**Verdict:** {'✅ APPROVED' if verdict == 'APPROVE' else '❌ CHANGES REQUESTED'}")
    lines.append("")
    lines.append(f"> {overall}")
    lines.append("")
    lines.append("---")
    lines.append("")

    dim_names = {
        "architecture": "🏛️ Architecture",
        "security": "🛡️ Security",
        "code_quality": "🔍 Code Quality",
        "tests": "🧪 Tests",
        "documentation": "📝 Documentation",
    }

    for key, label in dim_names.items():
        dim = dims.get(key, {})
        rating = dim.get("rating", "PASS")
        summary = dim.get("summary", "")
        findings = dim.get("findings", [])

        emoji = rating_emoji.get(rating, "ℹ️")
        lines.append(f"### {label} {emoji} `{rating}`")
        lines.append(f"_{summary}_")

        if findings:
            lines.append("")
            for f in findings:
                sev = f.get("severity", "info")
                sev_icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(sev, "🔵")
                file_ref = f.get("file", "")
                line_ref = f.get("line")
                loc = f"`{file_ref}:{line_ref}`" if line_ref else f"`{file_ref}`"
                msg = f.get("message", "")
                lines.append(f"- {sev_icon} {loc} — {msg}")

        lines.append("")

    lines.append("---")
    lines.append(
        "_Review generated by [AI PR Reviewer](/.github/scripts/ai_pr_reviewer.py) "
        f"using GitHub Models · `{MODEL}`. Verify all findings before merging._"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AI PR Reviewer for tool-selector-cascade")
    parser.add_argument("--pr", required=True, type=int, help="PR number")
    parser.add_argument("--repo", required=True, help="GitHub repo (owner/name)")
    args = parser.parse_args()

    github_token = os.environ.get("GITHUB_TOKEN", "")

    if not github_token:
        print("[ERROR] GITHUB_TOKEN is not set.")
        sys.exit(1)

    print(f"[AI Review] Fetching PR #{args.pr} from {args.repo}...")
    pr_meta = fetch_pr_metadata(args.repo, args.pr, github_token)
    diff = fetch_pr_diff(args.repo, args.pr, github_token)
    files = fetch_pr_files(args.repo, args.pr, github_token)

    print(f"[AI Review] PR: {pr_meta.get('title')} — {len(diff)} diff chars, {len(files)} files")

    user_message = build_user_message(pr_meta, diff, files)

    print(f"[AI Review] Calling GitHub Models ({MODEL})...")
    try:
        review = call_github_model(user_message, github_token)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Model returned invalid JSON: {exc}")
        sys.exit(1)

    verdict = review.get("verdict", "COMMENT")
    github_action = "APPROVE" if verdict == "APPROVE" else "REQUEST_CHANGES"

    review_body = format_github_review(review)
    print(review_body)

    post_pr_review(args.repo, args.pr, github_token, review_body, github_action)
    print(f"[AI Review] Done. Verdict: {github_action}")


if __name__ == "__main__":
    main()
