"""OpenAI-compatible facade that forwards Hermes turns to `gemini --acp`.

Unlike the old custom-provider gateway approach, this client treats Gemini CLI
as an ACP runtime. Hermes still owns the conversation, but Gemini receives a
native MCP server that exposes Hermes tools directly, so tool execution no
longer depends on fake OpenAI tool_calls.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ACP_MARKER_BASE_URL = "acp://gemini"
_DEFAULT_TIMEOUT_SECONDS = 900.0


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
    return os.getenv("HERMES_TOOLS_MCP_COMMAND", "").strip() or os.path.realpath(
        os.getenv("HERMES_PYTHON", "") or os.sys.executable
    )


def _tools_mcp_args() -> list[str]:
    raw = os.getenv("HERMES_TOOLS_MCP_ARGS", "").strip()
    if raw:
        return shlex.split(raw)
    repo_root = Path(__file__).resolve().parents[1]
    return [str(repo_root / "hermes_tools_mcp.py")]


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
        self._transcript: list[dict[str, Any]] = []

    def close(self) -> None:
        with self._process_lock:
            proc = self._process
            self._process = None
            self._session_id = None
            self._session_model = None
            self._session_tool_names = ()
            self._transcript = []
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
        env["NO_COLOR"] = "1"
        env["TERM"] = "dumb"
        return env

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
        self._transcript = []
        self._ensure_session(model=model, tool_names=tool_names, timeout_seconds=timeout_seconds)

    def _mcp_servers_payload(self, tool_names: tuple[str, ...]) -> list[dict[str, Any]]:
        env_entries = []
        include_value = ",".join(tool_names)
        env_entries.append({"name": "HERMES_TOOLS_MCP_INCLUDE", "value": include_value})
        hermes_home = os.getenv("HERMES_HOME", "").strip()
        if hermes_home:
            env_entries.append({"name": "HERMES_HOME", "value": hermes_home})
        return [
            {
                "name": "hermes-tools",
                "command": _tools_mcp_command(),
                "args": _tools_mcp_args(),
                "env": env_entries,
            }
        ]

    def _ensure_session(self, *, model: str | None, tool_names: tuple[str, ...], timeout_seconds: float) -> None:
        if (
            self._session_id
            and self._session_tool_names == tool_names
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
                    text_parts.append(text)
            elif kind == "agent_thought_chunk" and isinstance(content, dict) and reasoning_parts is not None:
                text = str(content.get("text") or "")
                if text:
                    reasoning_parts.append(text)
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
