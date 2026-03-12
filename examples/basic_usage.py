"""Basic usage examples for tool-selector-cascade.

Run this script (after installing the package with ``pip install -e .[all]``)
to see the cascade in action with mock tools.

Usage::

    cd tool_selector_cascade
    pip install -e .[all]
    python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)


@dataclass
class MockTool:
    """Minimal tool representation compatible with tool_selector_cascade."""

    name: str
    description: str


# Build a diverse mock tool pool (normally this would be your real registry)
ALL_TOOLS = [
    MockTool("gmail_send_email", "Send an email via Gmail"),
    MockTool("gmail_read_inbox", "Read emails from Gmail inbox"),
    MockTool("gmail_search", "Search Gmail by subject or sender"),
    MockTool("web_search", "Search the web for information"),
    MockTool("http_request", "Make an HTTP request to any URL"),
    MockTool("browser_navigate", "Navigate a browser to a URL"),
    MockTool("calendar_create_event", "Create a Google Calendar event"),
    MockTool("calendar_list_events", "List upcoming calendar events"),
    MockTool("calendar_delete_event", "Delete a calendar event by ID"),
    MockTool("slack_send_message", "Send a Slack channel message"),
    MockTool("slack_list_channels", "List Slack workspace channels"),
    MockTool("notion_create_page", "Create a Notion page"),
    MockTool("notion_search", "Search Notion workspace"),
    MockTool("jira_create_issue", "Create a Jira issue"),
    MockTool("jira_update_issue", "Update an existing Jira issue"),
    MockTool("github_create_pr", "Open a GitHub Pull Request"),
    MockTool("github_list_issues", "List GitHub repository issues"),
    MockTool("sql_query", "Execute a SQL query on a database"),
    MockTool("file_read", "Read a file from the filesystem"),
    MockTool("file_write", "Write content to a file"),
    MockTool("translate_text", "Translate text to another language"),
    MockTool("summarize_text", "Summarize a long document"),
    MockTool("weather_get", "Get current weather for a city"),
    MockTool("currency_convert", "Convert between currencies"),
    MockTool("image_analyze", "Analyze the content of an image"),
]


def demo_sync() -> None:
    """Demonstrate synchronous Level 1 + 2 selection."""
    from tool_selector_cascade import CascadeConfig, CascadeSelector

    print("\n=== Synchronous selection (Level 1 + 2) ===")

    selector = CascadeSelector(
        CascadeConfig(
            llm_enabled=False,  # sync API -- no LLM cost
            always_include_prefixes=["web_search"],
        )
    )

    # Pre-load models (would normally run in a background thread at startup)
    print("Loading models...")
    selector.warm_up(ALL_TOOLS)

    intent = "send an email to the team about the project deadline"
    print(f"Intent: {intent!r}")

    candidates, metrics = selector.select(intent, ALL_TOOLS, top_k=5)

    print(f"Selected {len(candidates)} tools:")
    for i, tool in enumerate(candidates):
        print(f"  [{i}] {tool.name}: {tool.description}")
    print(f"Metrics: {metrics.as_dict()}")


async def demo_async() -> None:
    """Demonstrate the full 3-level async cascade (requires API key)."""
    from tool_selector_cascade import CascadeConfig, CascadeSelector

    print("\n=== Async full cascade (Level 1 + 2 + 3) ===")
    print("Note: Level 3 will be skipped if ANTHROPIC_API_KEY is not set.")

    selector = CascadeSelector(
        CascadeConfig(
            llm_provider="anthropic",
            llm_model="claude-haiku-4-5",
            # api_key=None means it reads ANTHROPIC_API_KEY from environment
        )
    )

    selector.warm_up(ALL_TOOLS)

    intent = "schedule a meeting for next week"
    print(f"Intent: {intent!r}")

    result, metrics = await selector.aselect(intent, ALL_TOOLS)

    print(f"Final result ({len(result)} tool(s)):")
    for tool in result:
        print(f"  -> {tool.name}: {tool.description}")
    print(f"Metrics: {metrics.as_dict()}")


def demo_drop_in_replacement() -> None:
    """Demonstrate the select_tools_for_intent convenience function."""
    from tool_selector_cascade import select_tools_for_intent

    print("\n=== Drop-in replacement shim ===")

    intent = "search for information online"
    result = select_tools_for_intent(
        intent,
        ALL_TOOLS,
        top_k=5,
        always_include_prefixes=["web_search", "http_request"],
    )

    print(f"Intent: {intent!r}")
    print(f"Selected {len(result)} tools:")
    for tool in result:
        print(f"  - {tool.name}")


if __name__ == "__main__":
    demo_drop_in_replacement()
    demo_sync()
    asyncio.run(demo_async())
