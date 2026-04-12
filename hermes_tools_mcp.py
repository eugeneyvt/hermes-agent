#!/usr/bin/env python3
"""Expose selected Hermes tools as an MCP stdio server.

This is intentionally separate from ``hermes mcp serve``.  The built-in server
exports Hermes conversations/messages; this one exports callable agent tools so
external runtimes such as Gemini CLI can use the same tool implementations
without going through OpenAI-style tool_calls.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import sys
import types
import uuid
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - runtime-only guard
    FastMCP = None  # type: ignore[assignment]


# Import for side effects so the global registry is populated with built-ins,
# plugin tools, and any user-installed tool modules.
import model_tools  # noqa: F401
from model_tools import get_tool_definitions
from tools.registry import registry


logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Default to a stable subset that works outside the full Hermes agent loop.
_DEFAULT_INCLUDE = (
    "read_file",
    "write_file",
    "patch",
    "search_files",
    "terminal",
    "web_search",
    "web_extract",
    "web_crawl",
    "todo",
    "send_message",
    "vision_analyze",
    "image_generate",
    "text_to_speech",
)

# Known session-bound tools that are poor fits for an external MCP bridge.
_DEFAULT_EXCLUDE = {
    "clarify",
    "delegate",
    "memory",
    "session_search",
}


def _read_csv_env(name: str) -> tuple[bool, set[str]]:
    raw = os.getenv(name)
    if raw is None:
        return False, set()
    value = str(raw).strip()
    if not value:
        return True, set()
    return True, {part.strip() for part in value.split(",") if part.strip()}


def _json_type_to_python(schema: dict[str, Any]) -> Any:
    """Best-effort annotation mapping for FastMCP introspection."""
    kind = schema.get("type")
    if kind == "string":
        return str
    if kind == "integer":
        return int
    if kind == "number":
        return float
    if kind == "boolean":
        return bool
    if kind == "array":
        return list[Any]
    if kind == "object":
        return dict[str, Any]
    if isinstance(kind, list):
        non_null = [k for k in kind if k != "null"]
        if len(non_null) == 1:
            return _json_type_to_python({"type": non_null[0]}) | None
    return Any


def _make_signature(parameters: dict[str, Any] | None) -> inspect.Signature:
    parameters = parameters or {}
    props = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    sig_params: list[inspect.Parameter] = []

    for raw_name, prop_schema in props.items():
        if not _IDENT_RE.match(raw_name):
            continue
        annotation = _json_type_to_python(prop_schema or {})
        default = inspect._empty if raw_name in required else None
        sig_params.append(
            inspect.Parameter(
                raw_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )
    return inspect.Signature(sig_params)


def _exportable_schemas() -> list[dict[str, Any]]:
    include_present, include = _read_csv_env("HERMES_TOOLS_MCP_INCLUDE")
    exclude = set(_DEFAULT_EXCLUDE)
    _, extra_exclude = _read_csv_env("HERMES_TOOLS_MCP_EXCLUDE")
    exclude.update(extra_exclude)
    if not include_present:
        include = set(_DEFAULT_INCLUDE)

    openai_defs = get_tool_definitions(quiet_mode=True)
    exported: list[dict[str, Any]] = []
    for tool_def in openai_defs:
        fn = (tool_def or {}).get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        if name.startswith("mcp_") or name in exclude:
            continue
        if name not in include:
            continue
        exported.append(fn)
    return exported


def _make_handler(tool_name: str, description: str, parameters: dict[str, Any] | None):
    def _handler(**kwargs: Any) -> str:
        task_id = f"mcp-{tool_name}-{uuid.uuid4().hex[:8]}"
        return registry.dispatch(tool_name, kwargs, task_id=task_id)

    _handler.__name__ = f"tool_{tool_name}"
    _handler.__qualname__ = _handler.__name__
    _handler.__doc__ = description or f"Hermes tool '{tool_name}'"
    _handler.__signature__ = _make_signature(parameters)  # type: ignore[attr-defined]
    annotations: dict[str, Any] = {}
    for param in _handler.__signature__.parameters.values():  # type: ignore[attr-defined]
        if param.annotation is not inspect._empty:
            annotations[param.name] = param.annotation
    annotations["return"] = str
    _handler.__annotations__ = annotations
    return _handler


def create_tools_mcp_server() -> "FastMCP":
    if FastMCP is None:  # pragma: no cover - runtime-only guard
        raise ImportError(
            "Hermes tools MCP server requires the 'mcp' package. "
            "Install with: pip install 'hermes-agent[mcp]'"
        )

    mcp = FastMCP(
        "hermes-tools",
        instructions=(
            "Hermes Agent tool bridge. These tools mirror Hermes agent tools "
            "for external runtimes such as Gemini CLI."
        ),
    )

    exported = _exportable_schemas()
    for fn_schema in exported:
        tool_name = str(fn_schema.get("name") or "").strip()
        if not tool_name:
            continue
        handler = _make_handler(
            tool_name,
            str(fn_schema.get("description") or ""),
            fn_schema.get("parameters"),
        )
        mcp.add_tool(
            handler,
            name=tool_name,
            description=str(fn_schema.get("description") or ""),
        )

    logger.info("Hermes tools MCP server exporting %d tool(s)", len(exported))
    return mcp


def run_tools_mcp_server(verbose: bool = False) -> None:
    if FastMCP is None:  # pragma: no cover - runtime-only guard
        print(
            "Error: MCP server requires the 'mcp' package.\n"
            "Install with: pip install 'hermes-agent[mcp]'",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        stream=sys.stderr,
    )
    server = create_tools_mcp_server()
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    run_tools_mcp_server(verbose="--verbose" in sys.argv)
