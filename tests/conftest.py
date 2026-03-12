"""Shared fixtures for tool_selector_cascade tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


def _make_tool(name: str, description: str = "") -> MagicMock:
    """Create a mock tool with .name and .description attributes."""
    t = MagicMock(spec_set=["name", "description"])
    t.name = name
    t.description = description
    return t


@pytest.fixture
def sample_tools() -> list[Any]:
    """A pool of 25 mock tools covering diverse categories."""
    return [
        _make_tool("gmail_send_email", "Send an email via Gmail"),
        _make_tool("gmail_read_inbox", "Read emails from Gmail inbox"),
        _make_tool("web_search", "Search the web for information"),
        _make_tool("http_request", "Make an HTTP request to any URL"),
        _make_tool("browser_navigate", "Navigate a browser to a URL"),
        _make_tool("calendar_create_event", "Create a calendar event"),
        _make_tool("calendar_list_events", "List upcoming calendar events"),
        _make_tool("slack_send_message", "Send a Slack channel message"),
        _make_tool("notion_create_page", "Create a page in Notion"),
        _make_tool("jira_create_issue", "Create a Jira issue"),
        _make_tool("jira_update_issue", "Update an existing Jira issue"),
        _make_tool("github_create_pr", "Open a Pull Request on GitHub"),
        _make_tool("github_list_issues", "List issues in a GitHub repository"),
        _make_tool("sql_query", "Execute a SQL query on a database"),
        _make_tool("file_read", "Read a file from the filesystem"),
        _make_tool("file_write", "Write content to a file"),
        _make_tool("pdf_extract", "Extract text from a PDF document"),
        _make_tool("excel_create", "Create an Excel spreadsheet"),
        _make_tool("word_create", "Create a Word document"),
        _make_tool("translate_text", "Translate text to another language"),
        _make_tool("summarize_text", "Summarize a long article or document"),
        _make_tool("code_execute", "Execute a Python code snippet safely"),
        _make_tool("image_analyze", "Analyze the contents of an image"),
        _make_tool("weather_get", "Get current weather for a city"),
        _make_tool("currency_convert", "Convert between currencies"),
    ]


@pytest.fixture
def email_intent() -> str:
    return "send an email to the team about the project update"


@pytest.fixture
def web_intent() -> str:
    return "search for recent Python async programming tutorials"


@pytest.fixture
def small_tools(sample_tools: list[Any]) -> list[Any]:
    """A pool of 3 tools -- below the min_pool_size threshold."""
    return sample_tools[:3]
