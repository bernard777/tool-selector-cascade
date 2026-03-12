"""Protocol definitions and duck-typing helpers for tool representations.

This module defines the interfaces that ``tool_selector_cascade`` uses to interact
with tool objects without imposing any concrete class dependency.  Both
LangChain ``StructuredTool`` objects and raw ``dict`` representations
are supported transparently, formalised through ``typing.Protocol``.

Supported representations
-------------------------
* **LangChain StructuredTool** -- objects with ``.name`` and ``.description``
  string attributes.
* **dict** -- ``{"name": ..., "description": ...}`` or OpenAI function schema
  ``{"function": {"name": ..., "description": ...}}``.
* **Any object** -- anything that exposes ``.name`` and ``.description`` string
  attributes (duck typing).

Examples
--------
>>> from tool_selector_cascade.types import tool_as_text
>>> tool_as_text({"name": "web_search", "description": "Search the web"})
'web_search: Search the web'
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolLike(Protocol):
    """Minimal protocol for tool objects that expose a name and description."""

    @property
    def name(self) -> str:
        """The tool's unique identifier."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        ...


def extract_tool_name(tool: Any) -> str:
    """Extract the tool name from any supported representation.

    Parameters
    ----------
    tool:
        A tool object, dict, or any object with a ``.name`` attribute.

    Returns
    -------
    str
        The extracted name, or an empty string if not found.
    """
    if isinstance(tool, dict):
        d: dict[str, Any] = tool
        return (
            d.get("name") or d.get("technical_name") or (d.get("function") or {}).get("name") or ""
        ) or ""
    return getattr(tool, "name", "") or ""


def extract_tool_description(tool: Any) -> str:
    """Extract the tool description from any supported representation.

    Parameters
    ----------
    tool:
        A tool object, dict, or any object with a ``.description`` attribute.

    Returns
    -------
    str
        The extracted description, or an empty string if not found.
    """
    if isinstance(tool, dict):
        d: dict[str, Any] = tool
        return (d.get("description") or (d.get("function") or {}).get("description") or "") or ""
    return getattr(tool, "description", "") or ""


def tool_as_text(tool: Any) -> str:
    """Return a single string representation of a tool suitable for encoding.

    The format is ``"name: description"`` (or just ``"name"`` if there is no
    description).  This is the text fed to bi-encoder and cross-encoder models.

    Parameters
    ----------
    tool:
        Any supported tool representation.

    Returns
    -------
    str
    """
    name = extract_tool_name(tool)
    desc = extract_tool_description(tool)
    if name and desc:
        return f"{name}: {desc}"
    return name or desc
