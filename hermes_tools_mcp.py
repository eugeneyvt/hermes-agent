#!/usr/bin/env python3
"""Expose selected Hermes tools as an MCP stdio server.

This is intentionally separate from ``hermes mcp serve``.  The built-in server
exports Hermes conversations/messages; this one exports callable agent tools so
external runtimes such as Gemini CLI can use the same tool implementations
without going through OpenAI-style tool_calls.
"""

from __future__ import annotations

import asyncio
import argparse
import inspect
import json
import logging
import os
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import Context, FastMCP
except ImportError:  # pragma: no cover - runtime-only guard
    FastMCP = None  # type: ignore[assignment]
    Context = Any  # type: ignore[assignment]


# Import for side effects so the global registry is populated with built-ins,
# plugin tools, and any user-installed tool modules.
import model_tools  # noqa: F401
from model_tools import get_tool_definitions
from tools.registry import registry
from agent.display import (
    _detect_tool_failure,
    build_tool_preview as _build_tool_preview,
)


logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOOL_EVENT_URL_HEADER = "X-Hermes-Tool-Event-Url"
_TOOL_EVENT_TOKEN_HEADER = "X-Hermes-Tool-Event-Token"
_TOOL_EVENT_SIGNAL_PREFIX = "HERMES_TOOL_EVENT "

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


def _read_bool_env(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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


def _read_request_header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    for candidate in (name, name.lower()):
        try:
            value = headers.get(candidate)
        except Exception:
            value = None
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _write_tool_event_signal(payload: dict[str, Any], *, token: str | None = None) -> None:
    envelope: dict[str, Any] = {
        "kind": "tool_event",
        "payload": payload,
    }
    clean_token = str(token or "").strip()
    if clean_token:
        envelope["token"] = clean_token
    try:
        sys.stderr.write(
            _TOOL_EVENT_SIGNAL_PREFIX
            + json.dumps(envelope, ensure_ascii=False, default=str)
            + "\n"
        )
        sys.stderr.flush()
    except Exception:
        logger.debug("Could not emit MCP tool event signal", exc_info=True)


class _ToolEventEmitter:
    def __init__(self, *, url: str | None = None, token: str | None = None) -> None:
        self.url = url
        self.token = token

    @classmethod
    def from_context(cls, ctx: Context | None) -> "_ToolEventEmitter":
        request_context = getattr(ctx, "request_context", None)
        request = getattr(request_context, "request", None)
        headers = getattr(request, "headers", None)
        return cls(
            url=_read_request_header(headers, _TOOL_EVENT_URL_HEADER),
            token=_read_request_header(headers, _TOOL_EVENT_TOKEN_HEADER),
        )

    def _post(self, payload: dict[str, Any]) -> None:
        if not self.url:
            return
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers[_TOOL_EVENT_TOKEN_HEADER] = self.token
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1.0):
                pass
        except Exception:
            logger.debug("Could not post MCP tool event", exc_info=True)

    def _emit(self, payload: dict[str, Any]) -> None:
        _write_tool_event_signal(payload, token=self.token)
        self._post(payload)

    def emit_started(self, tool_name: str, tool_call_id: str, args: dict[str, Any]) -> None:
        self._emit(
            {
                "event_type": "tool.started",
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "preview": _build_tool_preview(tool_name, args),
                "args": args,
            }
        )

    def emit_completed(
        self,
        tool_name: str,
        tool_call_id: str,
        args: dict[str, Any],
        result: str,
        duration: float,
        is_error: bool,
    ) -> None:
        self._emit(
            {
                "event_type": "tool.completed",
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "args": args,
                "result": result,
                "duration": duration,
                "is_error": is_error,
            }
        )


def _exportable_schemas() -> list[dict[str, Any]]:
    include_present, include = _read_csv_env("HERMES_TOOLS_MCP_INCLUDE")
    exclude = set(_DEFAULT_EXCLUDE)
    _, extra_exclude = _read_csv_env("HERMES_TOOLS_MCP_EXCLUDE")
    exclude.update(extra_exclude)
    allow_all = _read_bool_env("HERMES_TOOLS_MCP_ALLOW_ALL")
    if not include_present and not allow_all:
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
        if not allow_all and name not in include:
            continue
        exported.append(fn)
    return exported


def _make_handler(tool_name: str, description: str, parameters: dict[str, Any] | None):
    def _handler(ctx: Context | None = None, **kwargs: Any) -> str:
        task_id = f"mcp-{tool_name}-{uuid.uuid4().hex[:8]}"
        tool_call_id = f"mcp-call-{uuid.uuid4().hex[:12]}"
        emitter = _ToolEventEmitter.from_context(ctx)
        emitter.emit_started(tool_name, tool_call_id, kwargs)
        started_at = time.monotonic()
        try:
            result = registry.dispatch(tool_name, kwargs, task_id=task_id)
        except Exception as exc:
            result = f"Error executing tool '{tool_name}': {exc}"
            duration = time.monotonic() - started_at
            emitter.emit_completed(tool_name, tool_call_id, kwargs, result, duration, True)
            raise
        duration = time.monotonic() - started_at
        is_error, _ = _detect_tool_failure(tool_name, result)
        emitter.emit_completed(tool_name, tool_call_id, kwargs, result, duration, is_error)
        return result

    _handler.__name__ = f"tool_{tool_name}"
    _handler.__qualname__ = _handler.__name__
    _handler.__doc__ = description or f"Hermes tool '{tool_name}'"
    _handler.__signature__ = _make_signature(parameters)  # type: ignore[attr-defined]
    annotations: dict[str, Any] = {"ctx": Context | None}
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


def run_tools_mcp_server(
    *,
    verbose: bool = False,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
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
    if transport == "stdio":
        asyncio.run(server.run_stdio_async())
        return
    if transport == "streamable-http":
        server.settings.host = host
        server.settings.port = port
        asyncio.run(server.run_streamable_http_async())
        return
    raise ValueError(f"Unsupported transport: {transport}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expose Hermes tools as an MCP server.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport to expose.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for streamable-http transport.")
    parser.add_argument("--port", type=int, default=8000, help="Port for streamable-http transport.")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_tools_mcp_server(
        verbose=args.verbose,
        transport=args.transport,
        host=args.host,
        port=args.port,
    )
