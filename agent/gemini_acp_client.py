"""OpenAI-compatible facade that forwards Hermes turns to `gemini --acp`.

Unlike the old custom-provider gateway approach, this client treats Gemini CLI
as an ACP runtime. Hermes still owns the conversation, but Gemini receives a
native MCP server that exposes Hermes tools directly, so tool execution no
longer depends on fake OpenAI tool_calls.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import shlex
import socket
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ACP_MARKER_BASE_URL = "acp://gemini"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_MCP_HTTP_HOST = "127.0.0.1"
_DEFAULT_BUILTIN_TOOLS = ("google_web_search", "web_fetch")
_TOOL_EVENT_URL_HEADER = "X-Hermes-Tool-Event-Url"
_TOOL_EVENT_TOKEN_HEADER = "X-Hermes-Tool-Event-Token"
_TOOL_EVENT_SIGNAL_PREFIX = "HERMES_TOOL_EVENT "
_DEBUG_STREAM_ENV = "HERMES_GEMINI_ACP_DEBUG_STREAM"


logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    value = str(os.getenv(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_command() -> str:
    candidates = [
        os.getenv("HERMES_GEMINI_ACP_COMMAND", "").strip(),
        os.getenv("GEMINI_CLI_PATH", "").strip(),
        "gemini",
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return "gemini"


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_GEMINI_ACP_ARGS", "").strip()
    if not raw:
        return ["--acp"]
    return shlex.split(raw)


def _tools_mcp_command() -> str:
    override = os.getenv("HERMES_TOOLS_MCP_COMMAND", "").strip()
    if override:
        return override
    python_override = os.getenv("HERMES_PYTHON", "").strip()
    if python_override:
        return python_override
    # Keep the venv interpreter path intact. Resolving symlinks here collapses
    # virtualenv Python back to the system interpreter, which may not have the
    # MCP/runtime dependencies installed.
    return os.sys.executable


def _tools_mcp_args() -> list[str]:
    raw = os.getenv("HERMES_TOOLS_MCP_ARGS", "").strip()
    if raw:
        return shlex.split(raw)
    repo_root = Path(__file__).resolve().parents[1]
    return [str(repo_root / "hermes_tools_mcp.py")]


def _pick_free_port(host: str = _MCP_HTTP_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.05)
    return False


def _parse_tool_event_signal_line(line: str) -> tuple[str | None, dict[str, Any]] | None:
    text = str(line or "").strip()
    if not text.startswith(_TOOL_EVENT_SIGNAL_PREFIX):
        return None
    raw_payload = text[len(_TOOL_EVENT_SIGNAL_PREFIX):].strip()
    if not raw_payload:
        return None
    try:
        envelope = json.loads(raw_payload)
    except Exception:
        logger.debug("Could not parse Hermes MCP tool signal: %r", text, exc_info=True)
        return None
    if not isinstance(envelope, dict):
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return None
    token = str(envelope.get("token") or "").strip() or None
    return token, payload


class _PersistentToolsMCPHTTPServer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._url: str | None = None

    def close(self) -> None:
        with self._lock:
            proc = self._process
            stderr_thread = self._stderr_thread
            self._process = None
            self._stderr_thread = None
            self._url = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if stderr_thread is not None:
            try:
                stderr_thread.join(timeout=2)
            except Exception:
                pass

    def _handle_stderr_line(self, line: str) -> None:
        parsed = _parse_tool_event_signal_line(line)
        if parsed is not None:
            token, payload = parsed
            _TOOL_EVENT_RELAY.dispatch_signal(token, payload)
            return
        text = str(line or "").rstrip("\n")
        if text:
            logger.debug("Hermes tools MCP stderr: %s", text)

    def ensure_started(self, *, timeout_seconds: float) -> str:
        override_url = os.getenv("HERMES_TOOLS_MCP_URL", "").strip()
        if override_url:
            return override_url

        with self._lock:
            if self._process is not None and self._process.poll() is None and self._url:
                return self._url

            port = _pick_free_port()
            cmd = [
                _tools_mcp_command(),
                *_tools_mcp_args(),
                "--transport",
                "streamable-http",
                "--host",
                _MCP_HTTP_HOST,
                "--port",
                str(port),
            ]
            env = dict(os.environ)
            env["HERMES_TOOLS_MCP_ALLOW_ALL"] = "1"
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
            )
            stderr_thread: threading.Thread | None = None
            if proc.stderr is not None:
                def _stderr_reader() -> None:
                    for line in proc.stderr:
                        self._handle_stderr_line(line)

                stderr_thread = threading.Thread(
                    target=_stderr_reader,
                    daemon=True,
                )
                stderr_thread.start()
            url = f"http://{_MCP_HTTP_HOST}:{port}/mcp"
            self._process = proc
            self._stderr_thread = stderr_thread
            self._url = url

        if _wait_for_port(_MCP_HTTP_HOST, port, timeout_seconds):
            return url

        self.close()
        raise RuntimeError("Timed out waiting for Hermes tools MCP HTTP server to start.")


_TOOLS_MCP_HTTP_SERVER = _PersistentToolsMCPHTTPServer()
atexit.register(_TOOLS_MCP_HTTP_SERVER.close)


class _ToolEventRelayHTTPServer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None
        self._callbacks: dict[str, Any] = {}

    def close(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
            self._url = None
            self._callbacks = {}
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        if thread is not None:
            try:
                thread.join(timeout=2)
            except Exception:
                pass

    def ensure_started(self) -> str:
        with self._lock:
            if self._server is not None and self._url:
                return self._url

            owner = self

            class _Handler(BaseHTTPRequestHandler):
                def do_POST(self) -> None:  # noqa: N802
                    try:
                        length = int(self.headers.get("Content-Length") or "0")
                    except (TypeError, ValueError):
                        length = 0
                    raw_body = self.rfile.read(max(length, 0))
                    token = str(self.headers.get(_TOOL_EVENT_TOKEN_HEADER) or "").strip()
                    if not token:
                        self.send_response(403)
                        self.end_headers()
                        return
                    try:
                        payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                    except Exception:
                        self.send_response(400)
                        self.end_headers()
                        return
                    owner.dispatch(token, payload)
                    self.send_response(204)
                    self.end_headers()

                def log_message(self, format: str, *args: Any) -> None:
                    return

            server = ThreadingHTTPServer((_MCP_HTTP_HOST, 0), _Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = int(server.server_address[1])
            self._server = server
            self._thread = thread
            self._url = f"http://{_MCP_HTTP_HOST}:{port}/"
            return self._url

    def set_callback(self, token: str, callback: Any | None) -> None:
        if not token:
            return
        with self._lock:
            if callback is None:
                self._callbacks.pop(token, None)
            else:
                self._callbacks[token] = callback

    def dispatch(self, token: str, payload: dict[str, Any]) -> None:
        with self._lock:
            callback = self._callbacks.get(token)
        if callback is None:
            return
        try:
            callback(payload)
        except Exception:
            logger.debug("Gemini tool event relay callback failed", exc_info=True)

    def dispatch_signal(self, token: str | None, payload: dict[str, Any]) -> None:
        callbacks: list[Any]
        with self._lock:
            if token:
                callback = self._callbacks.get(token)
                callbacks = [callback] if callback is not None else []
            elif len(self._callbacks) == 1:
                callbacks = list(self._callbacks.values())
            else:
                callbacks = []
        for callback in callbacks:
            try:
                callback(payload)
            except Exception:
                logger.debug("Gemini tool event relay callback failed", exc_info=True)


_TOOL_EVENT_RELAY = _ToolEventRelayHTTPServer()
atexit.register(_TOOL_EVENT_RELAY.close)


def _default_system_prompt_path() -> Path:
    hermes_home = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
    path = hermes_home / "runtime" / "gemini-acp" / "system.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "\n".join(
                [
                    "# Hermes Gemini ACP Backend",
                    "",
                    "You are the active Gemini CLI ACP backend for Hermes Agent.",
                    "Use the available MCP tools when they help complete the task.",
                    "Do not mention Gemini CLI, ACP, MCP, or internal runtime details unless explicitly asked.",
                    "Do not emit OpenAI-style tool_calls or XML wrappers.",
                    "Answer normally after using tools.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return path


def _resolve_builtin_tools() -> tuple[str, ...] | None:
    raw = os.getenv("HERMES_GEMINI_ACP_BUILTIN_TOOLS", "").strip()
    if raw:
        if raw.lower() in {"*", "all"}:
            return None
        tools = tuple(part.strip() for part in raw.split(",") if part.strip())
        return tools or ()
    return _DEFAULT_BUILTIN_TOOLS


def _runtime_settings_path() -> Path | None:
    override = os.getenv("HERMES_GEMINI_ACP_SETTINGS_PATH", "").strip()
    if override:
        path = Path(override).expanduser()
    else:
        hermes_home = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
        path = hermes_home / "runtime" / "gemini-acp" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    builtin_tools = _resolve_builtin_tools()
    if builtin_tools is None:
        return None
    path.write_text(
        json.dumps({"tools": {"core": list(builtin_tools)}}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _message_key(message: dict[str, Any]) -> str:
    return json.dumps(
        {
            "role": message.get("role"),
            "content": message.get("content"),
            "name": message.get("name"),
            "tool_call_id": message.get("tool_call_id"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _messages_have_prefix(messages: list[dict[str, Any]], prefix: list[dict[str, Any]]) -> bool:
    if len(prefix) > len(messages):
        return False
    return all(_message_key(a) == _message_key(b) for a, b in zip(messages[: len(prefix)], prefix))


def _extract_tool_names(tools: list[dict[str, Any]] | None) -> tuple[str, ...]:
    names: list[str] = []
    for tool in tools or []:
        fn = (tool or {}).get("function") or {}
        name = str(fn.get("name") or "").strip()
        if name:
            names.append(name)
    return tuple(sorted(set(names)))


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    *,
    model: str | None,
    continuation: bool,
) -> str:
    sections: list[str] = [
        "You are serving as the active Gemini ACP backend for Hermes.",
        "Use the available MCP tools when they help.",
        "Return only the assistant response for Hermes.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")
    if continuation:
        sections.append("This prompt contains only new conversation turns since the previous call.")

    transcript: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "context"
        rendered = _render_message_content(message.get("content"))
        if not rendered:
            continue
        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }[role]
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    sections.append("Continue from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


class _ACPChatCompletions:
    def __init__(self, client: "GeminiACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "GeminiACPClient"):
        self.completions = _ACPChatCompletions(client)


class GeminiACPClient:
    """Minimal OpenAI-client-compatible facade for Gemini ACP."""

    supports_tool_signal_callback = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "gemini-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or _resolve_command()
        self._acp_args = list(acp_args or args or _resolve_args())
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self._system_prompt_path = _default_system_prompt_path()
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False

        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        self._inbox: queue.Queue[dict[str, Any]] | None = None
        self._next_id = 0
        self._session_id: str | None = None
        self._session_model: str | None = None
        self._session_tool_names: tuple[str, ...] = ()
        self._session_tool_event_headers: tuple[tuple[str, str], ...] = ()
        self._transcript: list[dict[str, Any]] = []
        self._stream_delta_callback: Any = None
        self._reasoning_callback: Any = None
        self._tool_progress_callback: Any = None
        self._tool_start_callback: Any = None
        self._tool_complete_callback: Any = None
        self._tool_signal_callback: Any = None
        self._tool_event_url: str | None = None
        self._tool_event_token: str | None = None
        self._debug_stream_chunks = _env_flag(_DEBUG_STREAM_ENV)
        self._debug_stream_request_started_at = 0.0
        self._debug_stream_last_chunk_at = 0.0
        self._debug_stream_chunk_count = 0

    def set_stream_handlers(
        self,
        *,
        stream_delta_callback: Any = None,
        reasoning_callback: Any = None,
        tool_progress_callback: Any = None,
        tool_start_callback: Any = None,
        tool_complete_callback: Any = None,
        tool_signal_callback: Any = None,
    ) -> None:
        self._stream_delta_callback = stream_delta_callback
        self._reasoning_callback = reasoning_callback
        self._tool_progress_callback = tool_progress_callback
        self._tool_start_callback = tool_start_callback
        self._tool_complete_callback = tool_complete_callback
        self._tool_signal_callback = tool_signal_callback
        self._refresh_tool_event_subscription()

    def close(self) -> None:
        with self._process_lock:
            proc = self._process
            self._process = None
            self._session_id = None
            self._session_model = None
            self._session_tool_names = ()
            self._session_tool_event_headers = ()
            self._transcript = []
        if self._tool_event_token:
            _TOOL_EVENT_RELAY.set_callback(self._tool_event_token, None)
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> Any:
        message_list = list(messages or [])
        tool_names = _extract_tool_names(tools)
        timeout_seconds = float(timeout or _DEFAULT_TIMEOUT_SECONDS)

        self._ensure_connection(timeout_seconds=timeout_seconds)
        self._ensure_session(model=model, tool_names=tool_names, timeout_seconds=timeout_seconds)

        continuation = _messages_have_prefix(message_list, self._transcript)
        if continuation:
            delta_messages = message_list[len(self._transcript):]
        else:
            self._reset_session(model=model, tool_names=tool_names, timeout_seconds=timeout_seconds)
            delta_messages = message_list
            continuation = False

        prompt_text = _format_messages_as_prompt(
            delta_messages,
            model=model,
            continuation=continuation,
        )

        if self._debug_stream_chunks:
            now = time.monotonic()
            self._debug_stream_request_started_at = now
            self._debug_stream_last_chunk_at = now
            self._debug_stream_chunk_count = 0
            logger.info(
                "Gemini ACP stream debug: prompt start session=%s model=%s continuation=%s delta_messages=%d chars=%d",
                self._session_id or "?",
                model or self._session_model or "gemini-acp",
                continuation,
                len(delta_messages),
                len(prompt_text),
            )

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage_box: dict[str, Any] = {}
        result = self._request(
            "session/prompt",
            {
                "sessionId": self._session_id,
                "prompt": [{"type": "text", "text": prompt_text}],
                "messageId": str(uuid.uuid4()),
            },
            timeout_seconds=timeout_seconds,
            text_parts=text_parts,
            reasoning_parts=reasoning_parts,
            usage_box=usage_box,
        ) or {}

        usage_payload = result.get("usage") if isinstance(result, dict) else None
        if isinstance(usage_payload, dict):
            usage_box.update(usage_payload)

        response_text = "".join(text_parts).strip()
        reasoning_text = "".join(reasoning_parts).strip()
        self._transcript = [
            *message_list,
            {"role": "assistant", "content": response_text},
        ]

        usage = SimpleNamespace(
            prompt_tokens=int(usage_box.get("inputTokens") or 0),
            completion_tokens=int(usage_box.get("outputTokens") or 0),
            total_tokens=int(usage_box.get("totalTokens") or 0),
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=int(usage_box.get("cachedReadTokens") or 0)
            ),
        )
        assistant_message = SimpleNamespace(
            content=response_text,
            tool_calls=[],
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage, model=model or "gemini-acp")

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["GEMINI_SYSTEM_MD"] = str(self._system_prompt_path)
        settings_path = _runtime_settings_path()
        if settings_path is not None:
            env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = str(settings_path)
        else:
            env.pop("GEMINI_CLI_SYSTEM_SETTINGS_PATH", None)
        env["NO_COLOR"] = "1"
        env["TERM"] = "dumb"
        return env

    def _refresh_tool_event_subscription(self) -> None:
        wants_tool_events = any(
            cb is not None
            for cb in (
                self._tool_progress_callback,
                self._tool_start_callback,
                self._tool_complete_callback,
            )
        )
        if not wants_tool_events:
            if self._tool_event_token:
                _TOOL_EVENT_RELAY.set_callback(self._tool_event_token, None)
            self._tool_event_url = None
            return
        if self._tool_event_token is None:
            self._tool_event_token = uuid.uuid4().hex
        try:
            self._tool_event_url = _TOOL_EVENT_RELAY.ensure_started()
        except Exception:
            logger.debug("Could not start Gemini tool event relay", exc_info=True)
            self._tool_event_url = None
            return
        _TOOL_EVENT_RELAY.set_callback(self._tool_event_token, self._handle_tool_event)

    def _tool_event_headers(self) -> list[dict[str, str]]:
        if not self._tool_event_token or not self._tool_event_url:
            return []
        return [
            {"name": _TOOL_EVENT_URL_HEADER, "value": self._tool_event_url},
            {"name": _TOOL_EVENT_TOKEN_HEADER, "value": self._tool_event_token},
        ]

    def _handle_tool_event(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if self._tool_signal_callback is not None:
            try:
                self._tool_signal_callback(payload)
            except Exception:
                logger.debug("Gemini tool signal callback failed", exc_info=True)
            return
        event_type = str(payload.get("event_type") or "").strip()
        tool_name = str(payload.get("tool_name") or "").strip()
        tool_call_id = str(payload.get("tool_call_id") or "").strip()
        preview = payload.get("preview")
        args = payload.get("args")
        result = payload.get("result")
        if not isinstance(args, dict):
            args = {}
        if result is not None and not isinstance(result, str):
            try:
                result = json.dumps(result, ensure_ascii=False)
            except Exception:
                result = str(result)

        if event_type == "tool.started":
            if self._tool_progress_callback is not None:
                try:
                    self._tool_progress_callback(event_type, tool_name, preview, args)
                except Exception:
                    logger.debug("Gemini tool progress callback failed", exc_info=True)
            if self._tool_start_callback is not None and tool_call_id:
                try:
                    self._tool_start_callback(tool_call_id, tool_name, args)
                except Exception:
                    logger.debug("Gemini tool start callback failed", exc_info=True)
            return

        if event_type == "tool.completed":
            duration = payload.get("duration")
            try:
                duration = float(duration or 0)
            except (TypeError, ValueError):
                duration = 0.0
            is_error = bool(payload.get("is_error", False))
            if self._tool_progress_callback is not None:
                try:
                    self._tool_progress_callback(
                        event_type,
                        tool_name,
                        None,
                        None,
                        duration=duration,
                        is_error=is_error,
                    )
                except Exception:
                    logger.debug("Gemini tool completion callback failed", exc_info=True)
            if self._tool_complete_callback is not None and tool_call_id:
                try:
                    self._tool_complete_callback(tool_call_id, tool_name, args, result or "")
                except Exception:
                    logger.debug("Gemini tool complete callback failed", exc_info=True)

    def _ensure_connection(self, *, timeout_seconds: float) -> None:
        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                return
            try:
                proc = subprocess.Popen(
                    [self._acp_command] + self._acp_args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    cwd=self._acp_cwd,
                    env=self._build_env(),
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Could not start Gemini ACP command '{self._acp_command}'. "
                    "Install Gemini CLI or set HERMES_GEMINI_ACP_COMMAND/GEMINI_CLI_PATH."
                ) from exc

            if proc.stdin is None or proc.stdout is None:
                proc.kill()
                raise RuntimeError("Gemini ACP process did not expose stdin/stdout pipes.")

            inbox: queue.Queue[dict[str, Any]] = queue.Queue()

            def _stdout_reader() -> None:
                for line in proc.stdout:
                    try:
                        inbox.put(json.loads(line))
                    except Exception:
                        inbox.put({"raw": line.rstrip("\n")})

            threading.Thread(target=_stdout_reader, daemon=True).start()

            self._process = proc
            self._inbox = inbox
            self._next_id = 0
            self._session_id = None
            self._session_model = None
            self._session_tool_names = ()
            self._transcript = []

        self._request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "auth": {"terminal": False},
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {
                    "name": "hermes-agent",
                    "title": "Hermes Agent",
                    "version": "0.0.0",
                },
            },
            timeout_seconds=timeout_seconds,
        )

    def _reset_session(self, *, model: str | None, tool_names: tuple[str, ...], timeout_seconds: float) -> None:
        self._session_id = None
        self._session_model = None
        self._session_tool_names = ()
        self._session_tool_event_headers = ()
        self._transcript = []
        self._ensure_session(model=model, tool_names=tool_names, timeout_seconds=timeout_seconds)

    def _mcp_servers_payload(self, tool_names: tuple[str, ...]) -> list[dict[str, Any]]:
        http_url = _TOOLS_MCP_HTTP_SERVER.ensure_started(timeout_seconds=10.0)
        return [
            {
                "name": "hermes-tools",
                "type": "http",
                "url": http_url,
                "headers": self._tool_event_headers(),
                "includeTools": list(tool_names),
            }
        ]

    def _ensure_session(self, *, model: str | None, tool_names: tuple[str, ...], timeout_seconds: float) -> None:
        tool_event_headers = tuple(
            (str(item.get("name") or ""), str(item.get("value") or ""))
            for item in self._tool_event_headers()
        )
        if (
            self._session_id
            and self._session_tool_names == tool_names
            and self._session_tool_event_headers == tool_event_headers
            and (self._session_model == model or not model)
        ):
            return

        result = self._request(
            "session/new",
            {
                "cwd": self._acp_cwd,
                "mcpServers": self._mcp_servers_payload(tool_names),
            },
            timeout_seconds=timeout_seconds,
        ) or {}
        session_id = str(result.get("sessionId") or "").strip()
        if not session_id:
            raise RuntimeError("Gemini ACP did not return a sessionId.")
        self._session_id = session_id
        self._session_tool_names = tool_names
        self._session_tool_event_headers = tool_event_headers
        self._transcript = []

        if model:
            self._request(
                "session/set_model",
                {"sessionId": session_id, "modelId": model},
                timeout_seconds=timeout_seconds,
            )
            self._session_model = model
        else:
            current = ((result.get("models") or {}).get("currentModelId") if isinstance(result, dict) else None)
            self._session_model = str(current or "").strip() or None

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float,
        text_parts: list[str] | None = None,
        reasoning_parts: list[str] | None = None,
        usage_box: dict[str, Any] | None = None,
    ) -> Any:
        proc = self._process
        inbox = self._inbox
        if proc is None or proc.stdin is None or inbox is None:
            raise RuntimeError("Gemini ACP process is not running.")

        self._next_id += 1
        request_id = self._next_id
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}) + "\n")
        proc.stdin.flush()

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                msg = inbox.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._handle_server_message(
                msg,
                process=proc,
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
                usage_box=usage_box,
            ):
                continue

            if msg.get("id") != request_id:
                continue
            if "error" in msg:
                err = msg.get("error") or {}
                raise RuntimeError(err.get("message") or str(err))
            return msg.get("result")

        raise TimeoutError(f"Timed out waiting for Gemini ACP response to {method}.")

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
        usage_box: dict[str, Any] | None,
    ) -> bool:
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            kind = str(update.get("sessionUpdate") or "").strip()
            content = update.get("content") or {}
            if kind == "agent_message_chunk" and isinstance(content, dict) and text_parts is not None:
                text = str(content.get("text") or "")
                if text:
                    if self._debug_stream_chunks:
                        now = time.monotonic()
                        delta_since_last = now - (self._debug_stream_last_chunk_at or now)
                        delta_since_start = now - (self._debug_stream_request_started_at or now)
                        self._debug_stream_last_chunk_at = now
                        self._debug_stream_chunk_count += 1
                        logger.info(
                            "Gemini ACP stream debug: chunk=%d session=%s +%.2fs total=%.2fs chars=%d preview=%r",
                            self._debug_stream_chunk_count,
                            self._session_id or "?",
                            delta_since_last,
                            delta_since_start,
                            len(text),
                            text[:80],
                        )
                    text_parts.append(text)
                    cb = self._stream_delta_callback
                    if cb is not None:
                        try:
                            cb(text)
                        except Exception:
                            pass
            elif kind == "agent_thought_chunk" and isinstance(content, dict) and reasoning_parts is not None:
                text = str(content.get("text") or "")
                if text:
                    reasoning_parts.append(text)
                    cb = self._reasoning_callback
                    if cb is not None:
                        try:
                            cb(text)
                        except Exception:
                            pass
            elif kind == "usage_update" and usage_box is not None:
                usage_box.update(update)
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}
        if method == "session/request_permission":
            options = params.get("options") or []
            selected = None
            for option in options:
                if isinstance(option, dict) and option.get("kind") == "allow_once":
                    selected = option.get("optionId")
                    break
            if selected is None and options:
                first = options[0]
                if isinstance(first, dict):
                    selected = first.get("optionId")
            if selected is None:
                response = _jsonrpc_error(message_id, -32602, "No permission options provided.")
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "outcome": {
                            "outcome": "selected",
                            "optionId": selected,
                        }
                    },
                }
        else:
            response = _jsonrpc_error(message_id, -32601, f"Unsupported Gemini ACP client method '{method}'.")

        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True
