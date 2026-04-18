from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import shlex
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import registry, tool_error

from .config import (
    DEFAULT_PREFETCH_TIMEOUT_S,
    DEFAULT_RECALL_LIMIT,
    DEFAULT_SEARCH_TIMEOUT_S,
    DEFAULT_SESSION_SYNC_TIMEOUT_S,
    as_bool,
    default_cli_command,
    load_provider_config,
    save_provider_config,
)

logger = logging.getLogger(__name__)


class MempalaceMemoryProvider(MemoryProvider):
    _MAX_PREFETCH_CACHE_ENTRIES = 16
    _QUEUE_JOIN_TIMEOUT = 10.0
    _QUEUE_PUT_TIMEOUT = 0.1
    _DEFAULT_QUEUE_MAXSIZE = 32

    _MCP_SERVER_NAME = "mempalace"
    _MCP_PREFIX = "mcp_mempalace_"
    _PUBLIC_PREFIX = "mempalace_"
    _MCP_STATUS_TOOL = "mempalace_status"
    _MCP_SEARCH_TOOL = "mempalace_search"
    _MCP_ADD_DRAWER_TOOL = "mempalace_add_drawer"
    _DEFAULT_DIRECT_WRITE_MAX_CHARS = 4000

    def __init__(self) -> None:
        self._hermes_home = os.path.expanduser("~/.hermes")
        self._config: dict[str, Any] = {}
        self._session_id = ""
        self._agent_name = "hermes"
        self._agent_identity = "hermes"
        self._workspace_name = "hermes"
        self._user_id = ""
        self._scope_tag = "default"
        self._turn_count = 0
        self._agent_context = "primary"
        self._writes_enabled = True

        self._command: list[str] = []
        self._resolved_command = ""
        self._command_cwd = ""
        self._transcript_export_dir = ""
        self._transcript_path = ""
        self._prefetch_cache: dict[str, str] = {}
        self._mcp_tool_schemas: list[dict[str, Any]] = []
        self._last_prefetch_status = "idle"
        self._last_warning = ""
        self._last_hook_status = "idle"
        self._last_search_status = "idle"
        self._last_mine_status = "idle"
        self._last_command = ""
        self._last_successful_write_at = ""
        self._last_failed_write_at = ""
        self._dropped_jobs = 0
        self._mcp_connected = False
        self._palace_healthy = False

        self._prefetch_lock = threading.Lock()
        self._transcript_lock = threading.Lock()
        self._fingerprint_lock = threading.Lock()
        self._session_message_fingerprints: set[str] = set()
        self._worker_queue: queue.Queue[Callable[[], None] | object] | None = None
        self._worker_thread: threading.Thread | None = None
        self._worker_sentinel = object()
        self._accepting_tasks = False
        self._queue_maxsize = self._DEFAULT_QUEUE_MAXSIZE

    @property
    def name(self) -> str:
        return "mempalace"

    def is_available(self) -> bool:
        if not self._config:
            self._config = load_provider_config(self._hermes_home)
        self._resolve_runtime()
        if not self._command:
            self._mcp_connected = False
            self._palace_healthy = False
            self._last_warning = "MemPalace command is not configured"
            return False
        try:
            self._ensure_mcp_connected()
            status_payload = self._mcp_result_value(
                self._call_mcp_tool(self._MCP_STATUS_TOOL, {}),
                default=None,
            )
            self._palace_healthy = self._status_looks_healthy(status_payload)
            if self._palace_healthy:
                self._last_warning = ""
            else:
                self._last_warning = f"MemPalace palace unhealthy: {status_payload}"
            return True
        except Exception as exc:
            self._mcp_connected = False
            self._palace_healthy = False
            self._last_warning = f"MemPalace availability check failed: {exc}"
            logger.debug("MemPalace provider unavailable", exc_info=True)
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home") or self._hermes_home
        self._config = load_provider_config(self._hermes_home)
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._writes_enabled = self._agent_context == "primary"
        self._agent_identity = self._normalize_name(
            kwargs.get("agent_identity") or self._config.get("agent_name") or "hermes"
        )
        self._workspace_name = self._normalize_name(kwargs.get("agent_workspace") or "hermes")
        raw_user_id = str(kwargs.get("user_id") or "").strip()
        self._user_id = self._normalize_name(raw_user_id) if raw_user_id else ""
        self._agent_name = self._normalize_name(
            self._config.get("agent_name") or self._agent_identity or "hermes"
        )
        self._scope_tag = self._build_scope_tag()
        self._turn_count = 0
        self._prefetch_cache = {}
        self._session_message_fingerprints = set()
        self._last_prefetch_status = "idle"
        self._last_warning = ""
        self._last_hook_status = "idle"
        self._last_search_status = "idle"
        self._last_mine_status = "idle"
        self._last_command = ""
        self._last_successful_write_at = ""
        self._last_failed_write_at = ""
        self._dropped_jobs = 0
        self._mcp_connected = False
        self._palace_healthy = False
        self._resolve_runtime()
        self._ensure_mcp_connected()
        self._transcript_export_dir = self._resolve_transcript_export_dir()
        Path(self._transcript_export_dir).mkdir(parents=True, exist_ok=True)
        self._transcript_path = str(Path(self._transcript_export_dir) / f"{self._session_id}.jsonl")
        self._ensure_transcript_initialized()
        self._start_worker()
        self._invoke_hook("session-start", synchronous=True)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = max(turn_number, 0)

    def system_prompt_block(self) -> str:
        lines = [
            "MemPalace memory provider is active via Hermes MCP.",
            f"- MCP server: {self._MCP_SERVER_NAME}",
            f"- CLI fallback command: {self._resolved_command or 'not configured'}",
            f"- Palace path: {self._palace_path() or 'default'}",
            f"- Scope tag: {self._scope_tag}",
            f"- Agent context: {self._agent_context}",
            f"- Transcript path: {self._transcript_path or 'not initialized'}",
            "- Treat recalled MemPalace context as background memory, not as fresh user input.",
            "- Use mempalace_status at session start for protocol/status context.",
            "- Use mempalace_search and the MemPalace KG tools before answering questions about past work, people, or decisions.",
            "- Session filing follows the upstream hook and conversation-mining flow, not direct Chroma writes from Hermes.",
        ]
        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        cache_key = self._prefetch_cache_key(query, session_id=session_id)
        with self._prefetch_lock:
            cached = self._prefetch_cache.pop(cache_key, "")
            if cached:
                self._last_prefetch_status = "cache_hit"
                return cached

        try:
            if self._enable_wakeup() and self._turn_count <= 1:
                result = self._run_wakeup()
                self._last_prefetch_status = "wake_up"
                return result

            result = self._run_search(query, limit=self._recall_limit())
            self._last_prefetch_status = "search_inline"
            return result
        except Exception as exc:
            self._last_prefetch_status = "error"
            self._record_warning(f"Prefetch failed: {exc}")
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        clean_query = (query or "").strip()
        if not clean_query:
            return
        cache_key = self._prefetch_cache_key(clean_query, session_id=session_id)
        with self._prefetch_lock:
            if cache_key in self._prefetch_cache:
                self._last_prefetch_status = "cache_primed"
                return

        def _work() -> None:
            try:
                block = self._run_search(clean_query, limit=self._recall_limit())
            except Exception as exc:
                self._record_warning(f"Background prefetch failed: {exc}")
                self._last_prefetch_status = "prefetch_error"
                return
            if not block:
                self._last_prefetch_status = "prefetched_empty"
                return
            with self._prefetch_lock:
                self._prefetch_cache[cache_key] = block
                self._trim_prefetch_cache()
                self._last_prefetch_status = "prefetched"

        self._enqueue_task(_work, label="prefetch")

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._writes_enabled:
            logger.debug(
                "MemPalace sync_turn skipped for non-primary context '%s'",
                self._agent_context,
            )
            return
        active_session = session_id or self._session_id
        clean_user = (user_content or "").strip()
        clean_assistant = (assistant_content or "").strip()
        if not clean_user or not clean_assistant:
            return
        self._ensure_transcript_initialized()
        self._append_transcript_event(
            {"type": "event_msg", "payload": {"type": "user_message", "message": clean_user}}
        )
        self._append_transcript_event(
            {"type": "event_msg", "payload": {"type": "agent_message", "message": clean_assistant}}
        )
        with self._fingerprint_lock:
            self._session_message_fingerprints.add(self._message_fingerprint("user", clean_user))
            self._session_message_fingerprints.add(
                self._message_fingerprint("assistant", clean_assistant)
            )

        def _work() -> None:
            payload = self._invoke_hook("stop", synchronous=False)
            if payload.get("decision") == "block":
                self._run_mine(background=True, reason="stop")
            self._mark_write_success(active_session)

        self._enqueue_task(_work, label="sync_turn")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._writes_enabled:
            logger.debug(
                "MemPalace on_session_end skipped for non-primary context '%s'",
                self._agent_context,
            )
            return

        for msg in messages or []:
            role = (msg or {}).get("role", "")
            if role not in ("user", "assistant"):
                continue
            clean = self._message_text(msg)
            if not clean:
                continue
            fingerprint = self._message_fingerprint(role, clean)
            with self._fingerprint_lock:
                if fingerprint in self._session_message_fingerprints:
                    continue
                self._session_message_fingerprints.add(fingerprint)
            payload_type = "user_message" if role == "user" else "agent_message"
            self._append_transcript_event(
                {"type": "event_msg", "payload": {"type": payload_type, "message": clean}}
            )

        self._invoke_hook("stop", synchronous=True)
        self._run_mine(background=False, reason="session_end")
        self._maybe_reconnect_mcp()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._writes_enabled:
            logger.debug(
                "MemPalace on_pre_compress skipped for non-primary context '%s'",
                self._agent_context,
            )
            return ""
        self._invoke_hook("precompact", synchronous=True)
        self._run_mine(background=False, reason="precompact")
        self._maybe_reconnect_mcp()
        return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._writes_enabled:
            logger.debug(
                "MemPalace on_memory_write skipped for non-primary context '%s'",
                self._agent_context,
            )
            return
        if action not in ("add", "replace"):
            self._record_warning(
                f"MemPalace does not mirror memory action '{action}' for target '{target}'"
            )
            return
        clean_content = (content or "").strip()
        if not clean_content:
            return

        room = self._user_room() if target == "user" else self._memory_room()
        payload = {
            "kind": "hermes_memory",
            "action": action,
            "target": target,
            "content": self._truncate_text(clean_content),
            "scope_tag": self._scope_tag,
            "session_id": self._session_id,
            "agent_identity": self._agent_identity,
            "agent_name": self._agent_name,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._enqueue_direct_write(
            wing=self._memory_wing(),
            room=room,
            payload=payload,
            label=f"memory_write:{target}",
        )

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs
    ) -> None:
        del kwargs
        if not self._writes_enabled:
            logger.debug(
                "MemPalace on_delegation skipped for non-primary context '%s'",
                self._agent_context,
            )
            return
        clean_task = (task or "").strip()
        clean_result = (result or "").strip()
        if not clean_task and not clean_result:
            return

        payload = {
            "kind": "hermes_delegation",
            "task": self._truncate_text(clean_task),
            "result": self._truncate_text(clean_result),
            "child_session_id": child_session_id,
            "scope_tag": self._scope_tag,
            "session_id": self._session_id,
            "agent_identity": self._agent_identity,
            "agent_name": self._agent_name,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._enqueue_direct_write(
            wing=self._memory_wing(),
            room=self._delegation_room(),
            payload=payload,
            label="delegation",
        )

    def shutdown(self) -> None:
        self._stop_worker()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._mcp_tool_schemas:
            return list(self._mcp_tool_schemas)
        if self._config:
            try:
                self._ensure_mcp_connected()
            except Exception:
                logger.debug("MemPalace MCP schemas unavailable", exc_info=True)
        return list(self._mcp_tool_schemas)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        actual_name = self._resolve_tool_name(tool_name)
        if actual_name is None:
            return tool_error(f"Unknown MemPalace MCP tool: {tool_name}")
        try:
            self._ensure_mcp_connected()
            result = self._call_mcp_tool(actual_name, args)
            if actual_name == self._MCP_STATUS_TOOL:
                result = self._enrich_status_result(result)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            logger.warning("MemPalace MCP tool failed: %s", exc, exc_info=True)
            return tool_error(f"MemPalace MCP tool failed: {exc}")

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "repo_path",
                "description": "Optional local MemPalace checkout. Preferred for auto-managed MCP launch.",
                "required": False,
                "default": "",
            },
            {
                "key": "command",
                "description": "Command used to run MemPalace CLI fallback operations (for example 'mempalace' or 'python3 -m mempalace')",
                "required": False,
                "default": default_cli_command(),
            },
            {
                "key": "palace_path",
                "description": "Path to the MemPalace data directory",
                "required": False,
                "default": os.path.expanduser("~/.mempalace/palace"),
            },
            {
                "key": "transcript_export_dir",
                "description": "Directory where Hermes writes Codex-compatible session transcripts for MemPalace hooks",
                "required": False,
                "default": "",
            },
            {
                "key": "enable_wakeup",
                "description": "Whether to use CLI 'mempalace wake-up' for first-turn context injection",
                "required": False,
                "default": True,
            },
            {
                "key": "recall_limit",
                "description": "Default number of search results requested from MemPalace",
                "required": False,
                "default": DEFAULT_RECALL_LIMIT,
            },
            {
                "key": "search_timeout_s",
                "description": "Timeout for search and wake-up CLI commands",
                "required": False,
                "default": DEFAULT_SEARCH_TIMEOUT_S,
            },
            {
                "key": "session_sync_timeout_s",
                "description": "Timeout for synchronous session save and mine operations",
                "required": False,
                "default": DEFAULT_SESSION_SYNC_TIMEOUT_S,
            },
            {
                "key": "prefetch_timeout_s",
                "description": "Timeout for background prefetch search operations",
                "required": False,
                "default": DEFAULT_PREFETCH_TIMEOUT_S,
            },
            {
                "key": "queue_maxsize",
                "description": "Maximum number of background MemPalace jobs queued by Hermes",
                "required": False,
                "default": self._DEFAULT_QUEUE_MAXSIZE,
            },
            {
                "key": "agent_name",
                "description": "Logical agent/profile name used for transcript metadata",
                "required": False,
                "default": "hermes",
            },
            {
                "key": "scope_by_profile",
                "description": "Whether to isolate memory by Hermes profile name",
                "required": False,
                "default": True,
            },
            {
                "key": "scope_by_user",
                "description": "Whether to isolate memory by gateway user_id when present",
                "required": False,
                "default": True,
            },
            {
                "key": "memory_wing",
                "description": "Base wing name for Hermes direct-write durable facts and delegation records",
                "required": False,
                "default": "wing_hermes_memory",
            },
            {
                "key": "memory_room",
                "description": "Room used for built-in MEMORY.md fact mirrors",
                "required": False,
                "default": "memory",
            },
            {
                "key": "user_room",
                "description": "Room used for built-in USER.md fact mirrors",
                "required": False,
                "default": "user",
            },
            {
                "key": "delegation_room",
                "description": "Room used for parent-side delegation summaries",
                "required": False,
                "default": "delegations",
            },
            {
                "key": "direct_write_max_chars",
                "description": "Maximum characters persisted for direct-write fact and delegation mirrors",
                "required": False,
                "default": self._DEFAULT_DIRECT_WRITE_MAX_CHARS,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        existing = load_provider_config(hermes_home)
        sanitized = {
            "repo_path": str(values.get("repo_path") or "").strip(),
            "command": str(values.get("command") or "").strip(),
            "palace_path": str(values.get("palace_path") or "").strip(),
            "transcript_export_dir": str(values.get("transcript_export_dir") or "").strip(),
            "enable_wakeup": as_bool(values.get("enable_wakeup"), True),
            "recall_limit": int(values.get("recall_limit") or DEFAULT_RECALL_LIMIT),
            "search_timeout_s": float(values.get("search_timeout_s") or DEFAULT_SEARCH_TIMEOUT_S),
            "session_sync_timeout_s": float(
                values.get("session_sync_timeout_s") or DEFAULT_SESSION_SYNC_TIMEOUT_S
            ),
            "prefetch_timeout_s": float(
                values.get("prefetch_timeout_s") or DEFAULT_PREFETCH_TIMEOUT_S
            ),
            "queue_maxsize": max(1, int(values.get("queue_maxsize") or self._DEFAULT_QUEUE_MAXSIZE)),
            "agent_name": self._normalize_name(values.get("agent_name") or "hermes"),
            "scope_by_profile": as_bool(values.get("scope_by_profile"), True),
            "scope_by_user": as_bool(values.get("scope_by_user"), True),
            "memory_wing": self._normalize_name(
                values.get("memory_wing") or existing.get("memory_wing") or "wing_hermes_memory"
            ),
            "memory_room": self._normalize_name(
                values.get("memory_room") or existing.get("memory_room") or "memory"
            ),
            "user_room": self._normalize_name(
                values.get("user_room") or existing.get("user_room") or "user"
            ),
            "delegation_room": self._normalize_name(
                values.get("delegation_room") or existing.get("delegation_room") or "delegations"
            ),
            "direct_write_max_chars": max(
                256,
                int(
                    values.get("direct_write_max_chars")
                    or existing.get("direct_write_max_chars")
                    or self._DEFAULT_DIRECT_WRITE_MAX_CHARS
                ),
            ),
        }
        merged = existing
        merged.update(sanitized)
        save_provider_config(merged, hermes_home)

        from hermes_cli.config import load_config, save_config

        config = load_config()
        config.setdefault("mcp_servers", {})[self._MCP_SERVER_NAME] = self._build_mcp_server_config(
            sanitized
        )
        save_config(config)

    def _normalize_name(self, value: str) -> str:
        text = (value or "default").strip().lower()
        cleaned = []
        for char in text:
            if char.isalnum() or char in "._-":
                cleaned.append(char)
            elif char.isspace():
                cleaned.append("_")
            else:
                cleaned.append("-")
        normalized = "".join(cleaned).strip("._- ")
        return normalized or "default"

    def _build_scope_tag(self) -> str:
        parts = []
        if as_bool(self._config.get("scope_by_profile"), True):
            parts.append(f"profile_{self._agent_identity}")
        if as_bool(self._config.get("scope_by_user"), True) and self._user_id:
            parts.append(f"user_{self._user_id}")
        if not parts:
            parts.append("shared")
        return self._normalize_name("__".join(parts))

    def _resolve_runtime(self) -> None:
        command = str(self._config.get("command") or "").strip()
        repo_path = str(self._config.get("repo_path") or "").strip()
        repo_path = os.path.abspath(os.path.expanduser(repo_path)) if repo_path else ""
        command_cwd = ""

        if command:
            command_parts = shlex.split(command)
        else:
            venv_cmd = Path(repo_path) / ".venv" / "bin" / "mempalace" if repo_path else None
            if venv_cmd and venv_cmd.exists():
                command_parts = [str(venv_cmd)]
            elif repo_path and shutil.which("uv"):
                command_parts = ["uv", "run", "--project", repo_path, "mempalace"]
            else:
                fallback = default_cli_command()
                command_parts = shlex.split(fallback) if fallback else []
                if repo_path and command_parts[:3] == ["python3", "-m", "mempalace"]:
                    command_cwd = repo_path

        if command_parts and command_parts[0] == "mempalace":
            resolved = shutil.which(command_parts[0]) or command_parts[0]
            command_parts[0] = resolved

        self._command = command_parts
        self._resolved_command = " ".join(shlex.quote(part) for part in command_parts)
        self._command_cwd = command_cwd

    def _palace_path(self) -> str:
        return str(self._config.get("palace_path") or "").strip()

    def _resolve_transcript_export_dir(self) -> str:
        configured = str(self._config.get("transcript_export_dir") or "").strip()
        if configured:
            return os.path.abspath(os.path.expanduser(configured))
        return str(Path(self._hermes_home) / "mempalace_transcripts")

    def _build_mcp_server_config(self, cfg: Optional[Dict[str, Any]] = None) -> dict:
        cfg = cfg or self._config
        command, args = self._resolve_mcp_command(cfg)
        return {
            "command": command,
            "args": args,
            "enabled": True,
            "timeout": max(30, int(float(cfg.get("session_sync_timeout_s") or DEFAULT_SESSION_SYNC_TIMEOUT_S))),
            "connect_timeout": max(30, int(float(cfg.get("search_timeout_s") or DEFAULT_SEARCH_TIMEOUT_S))),
        }

    def _resolve_mcp_command(self, cfg: Dict[str, Any]) -> tuple[str, list[str]]:
        command_str = str(cfg.get("command") or "").strip()
        repo_path = str(cfg.get("repo_path") or "").strip()
        repo_path = os.path.abspath(os.path.expanduser(repo_path)) if repo_path else ""
        palace_path = str(cfg.get("palace_path") or "").strip()

        args: list[str]
        command: str

        if command_str:
            parts = shlex.split(command_str) if command_str else []
            python_cmd = shutil.which("python3") or shutil.which("python") or "python3"
            if parts and Path(parts[0]).name.startswith("python"):
                command = parts[0]
                tail = list(parts[1:])
                if "-m" in tail:
                    idx = tail.index("-m")
                    if idx + 1 < len(tail):
                        tail[idx + 1] = "mempalace.mcp_server"
                        args = tail
                    else:
                        args = tail + ["mempalace.mcp_server"]
                else:
                    args = tail + ["-m", "mempalace.mcp_server"]
            elif parts:
                inferred_python = self._infer_python_from_command(parts[0])
                if inferred_python:
                    command = inferred_python
                    args = ["-m", "mempalace.mcp_server"]
                else:
                    command = python_cmd
                    args = ["-m", "mempalace.mcp_server"]
            else:
                command = python_cmd
                args = ["-m", "mempalace.mcp_server"]
        else:
            venv_python = Path(repo_path) / ".venv" / "bin" / "python" if repo_path else None
            if venv_python and venv_python.exists():
                command = str(venv_python)
                args = ["-m", "mempalace.mcp_server"]
            elif repo_path and shutil.which("uv"):
                command = shutil.which("uv") or "uv"
                args = ["run", "--project", repo_path, "python", "-m", "mempalace.mcp_server"]
            else:
                python_cmd = shutil.which("python3") or shutil.which("python") or "python3"
                command = python_cmd
                args = ["-m", "mempalace.mcp_server"]

        if palace_path:
            args = list(args) + ["--palace", os.path.abspath(os.path.expanduser(palace_path))]
        return command, args

    @staticmethod
    def _infer_python_from_command(command: str) -> str:
        resolved = shutil.which(command) or command
        path = Path(os.path.expanduser(resolved))

        for candidate in (path.with_name("python"), path.with_name("python3")):
            if candidate.exists():
                return str(candidate)

        try:
            with open(path, "rb") as handle:
                first_line = handle.readline().decode("utf-8", errors="ignore").strip()
        except OSError:
            return ""

        if not first_line.startswith("#!"):
            return ""

        shebang_parts = shlex.split(first_line[2:].strip())
        if not shebang_parts:
            return ""
        if Path(shebang_parts[0]).name == "env" and len(shebang_parts) > 1:
            return shutil.which(shebang_parts[1]) or shebang_parts[1]
        return shebang_parts[0]

    def _ensure_mcp_connected(self) -> None:
        from tools.mcp_tool import (
            get_mcp_tool_schemas,
            is_mcp_server_connected,
            register_mcp_servers,
        )

        server_config = self._build_mcp_server_config()
        if not is_mcp_server_connected(self._MCP_SERVER_NAME):
            register_mcp_servers({self._MCP_SERVER_NAME: server_config})

        schemas = get_mcp_tool_schemas(self._MCP_SERVER_NAME)
        if not schemas:
            self._mcp_connected = False
            raise RuntimeError("MemPalace MCP server is not connected")

        self._mcp_connected = True
        self._mcp_tool_schemas = [
            self._public_tool_schema(schema)
            for schema in schemas
            if schema.get("name", "").startswith(self._MCP_PREFIX)
        ]
        if not self._mcp_tool_schemas:
            raise RuntimeError("MemPalace MCP server exposed no tool schemas")

    def _call_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> dict:
        from tools.mcp_tool import call_mcp_tool

        timeout = max(self._search_timeout(), self._session_sync_timeout())
        return call_mcp_tool(self._MCP_SERVER_NAME, tool_name, args, timeout=timeout)

    @staticmethod
    def _mcp_result_value(result: dict, *, default: Any = "") -> Any:
        if not isinstance(result, dict):
            return default
        if "structuredContent" in result:
            return result["structuredContent"]
        return result.get("result", default)

    @classmethod
    def _strip_mcp_prefix(cls, tool_name: str) -> Optional[str]:
        if not tool_name.startswith(cls._MCP_PREFIX):
            return None
        return tool_name[len(cls._MCP_PREFIX) :]

    @classmethod
    def _resolve_tool_name(cls, tool_name: str) -> Optional[str]:
        stripped = cls._strip_mcp_prefix(tool_name)
        if stripped is not None:
            return stripped
        if not tool_name:
            return None
        if tool_name.startswith(cls._PUBLIC_PREFIX):
            return tool_name
        return None

    @classmethod
    def _public_tool_schema(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        exposed = dict(schema)
        actual_name = cls._strip_mcp_prefix(schema.get("name", "") or "")
        if actual_name:
            exposed["name"] = actual_name
        return exposed

    def _enrich_status_result(self, result: dict) -> dict:
        payload = self._mcp_result_value(result, default={})
        normalized = self._normalize_status_payload(payload)
        self._palace_healthy = self._status_looks_healthy(normalized)
        runtime = self._provider_runtime_status()

        if isinstance(normalized, dict):
            enriched = dict(normalized)
            enriched["provider_runtime"] = runtime
        else:
            enriched = {"raw_status": normalized, "provider_runtime": runtime}

        if "result" in result:
            return {**result, "result": enriched}
        if "structuredContent" in result:
            return {**result, "structuredContent": enriched}
        return {"result": enriched}

    @staticmethod
    def _normalize_status_payload(payload: Any) -> Any:
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return payload

    @staticmethod
    def _status_looks_healthy(payload: Any) -> bool:
        if isinstance(payload, dict):
            if payload.get("error"):
                return False
            return True
        text = str(payload or "").strip().lower()
        if not text:
            return True
        unhealthy_markers = (
            "no palace found",
            "run: mempalace init",
            "not initialized",
        )
        return not any(marker in text for marker in unhealthy_markers)

    def _base_cli_command(self) -> list[str]:
        if not self._command:
            self._resolve_runtime()
        if not self._command:
            raise RuntimeError("MemPalace command is not configured")
        cmd = list(self._command)
        palace_path = self._palace_path()
        if palace_path:
            cmd.extend(["--palace", palace_path])
        return cmd

    def _run_cli(
        self,
        args: list[str],
        *,
        timeout: float,
        check: bool,
        stdin_json: dict[str, Any] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = self._base_cli_command() + args
        self._last_command = " ".join(shlex.quote(part) for part in command)
        payload = None
        if stdin_json is not None:
            payload = json.dumps(stdin_json, ensure_ascii=False)
        result = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=self._command_cwd or None,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if check and result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or f"exit code {result.returncode}"
            raise RuntimeError(detail)
        return result

    def _run_status(self) -> str:
        result = self._run_cli(["status"], timeout=self._status_timeout(), check=True)
        return (result.stdout or "").strip()

    def _run_wakeup(self) -> str:
        args = ["wake-up"]
        result = self._run_cli(args, timeout=self._search_timeout(), check=True)
        output = (result.stdout or "").strip()
        self._last_search_status = "wake_up_ok"
        return output

    def _run_search(
        self, query: str, *, limit: int, wing: str | None = None, room: str | None = None
    ) -> str:
        self._ensure_mcp_connected()
        mcp_result = self._call_mcp_tool(
            self._MCP_SEARCH_TOOL,
            {
                "query": query,
                "limit": limit,
                **({"wing": wing} if wing else {}),
                **({"room": room} if room else {}),
            },
        )
        if mcp_result.get("error"):
            raise RuntimeError(str(mcp_result["error"]))
        output = self._mcp_result_value(mcp_result, default="")
        self._last_search_status = "search_ok"
        if isinstance(output, str):
            return output.strip()
        return json.dumps(output, ensure_ascii=False, indent=2)

    def _run_mine(self, *, background: bool, reason: str) -> None:
        transcript_dir = self._transcript_export_dir
        args = [
            "mine",
            transcript_dir,
            "--mode",
            "convos",
            "--wing",
            self._conversation_wing(),
            "--agent",
            self._agent_name,
        ]
        timeout = self._session_sync_timeout()
        self._run_cli(args, timeout=timeout, check=True)
        if background:
            self._last_mine_status = f"background_{reason}_ok"
        else:
            self._last_mine_status = f"sync_{reason}_ok"

    def _invoke_hook(self, hook_name: str, *, synchronous: bool) -> dict[str, Any]:
        payload = {
            "session_id": self._session_id,
            "transcript_path": self._transcript_path,
            "stop_hook_active": False,
        }
        timeout = self._session_sync_timeout() if synchronous else self._prefetch_timeout()
        result = self._run_cli(
            ["hook", "run", "--hook", hook_name, "--harness", "codex"],
            timeout=timeout,
            check=True,
            stdin_json=payload,
        )
        output = (result.stdout or "").strip()
        self._last_hook_status = f"{hook_name}_ok"
        if not output:
            return {}
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"raw_output": output}

    def _maybe_reconnect_mcp(self) -> None:
        try:
            self._ensure_mcp_connected()
            self._call_mcp_tool("mempalace_reconnect", {})
        except Exception:
            logger.debug("MemPalace MCP reconnect skipped", exc_info=True)

    def _ensure_transcript_initialized(self) -> None:
        if not self._transcript_path:
            return
        transcript = Path(self._transcript_path)
        if transcript.exists():
            return
        meta = {
            "type": "session_meta",
            "payload": {
                "session_id": self._session_id,
                "agent_identity": self._agent_identity,
                "scope_tag": self._scope_tag,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")

    def _append_transcript_event(self, entry: dict[str, Any]) -> None:
        if not self._transcript_path:
            return
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._transcript_lock:
            with open(self._transcript_path, "a", encoding="utf-8") as handle:
                handle.write(line)

    def _message_text(self, message: Dict[str, Any]) -> str:
        content = (message or {}).get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "\n".join(part for part in text_parts if part)
        if not isinstance(content, str):
            return ""
        return content.strip()

    def _message_fingerprint(self, role: str, content: str) -> str:
        return f"{role}:{hash((content or '').strip())}"

    def _prefetch_cache_key(self, query: str, *, session_id: str = "") -> str:
        active_session = session_id or self._session_id or "default"
        return f"{active_session}:{hash((query or '').strip())}"

    def _conversation_wing(self) -> str:
        return self._scoped_wing("conversation_wing", "wing_hermes_sessions")

    def _memory_wing(self) -> str:
        return self._scoped_wing("memory_wing", "wing_hermes_memory")

    def _scoped_wing(self, config_key: str, default: str) -> str:
        base = self._normalize_name(str(self._config.get(config_key) or default).strip())
        if not base:
            base = default
        return f"{base}__{self._scope_tag}"

    def _memory_room(self) -> str:
        return self._normalize_name(str(self._config.get("memory_room") or "memory"))

    def _user_room(self) -> str:
        return self._normalize_name(str(self._config.get("user_room") or "user"))

    def _delegation_room(self) -> str:
        return self._normalize_name(str(self._config.get("delegation_room") or "delegations"))

    def _direct_write_max_chars(self) -> int:
        return max(
            256,
            int(self._config.get("direct_write_max_chars") or self._DEFAULT_DIRECT_WRITE_MAX_CHARS),
        )

    def _truncate_text(self, text: str) -> str:
        clean = (text or "").strip()
        limit = self._direct_write_max_chars()
        if len(clean) <= limit:
            return clean
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()[:12]
        return f"{clean[:limit]}\n...[truncated sha256:{digest}]"

    def _enqueue_direct_write(
        self, *, wing: str, room: str, payload: Dict[str, Any], label: str
    ) -> None:
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True)

        def _work() -> None:
            result = self._call_mcp_tool(
                self._MCP_ADD_DRAWER_TOOL,
                {
                    "wing": wing,
                    "room": room,
                    "content": content,
                    "added_by": "hermes",
                    "source_file": self._transcript_path or "",
                },
            )
            result_payload = self._mcp_result_value(result, default={})
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(str(result["error"]))
            if isinstance(result_payload, dict) and not result_payload.get("success", True):
                raise RuntimeError(result_payload.get("error") or f"Direct write failed for {label}")
            self._mark_write_success(self._session_id)

        self._enqueue_task(_work, label=label)

    def _provider_runtime_status(self) -> Dict[str, Any]:
        return {
            "mcp_connected": self._mcp_connected,
            "palace_healthy": self._palace_healthy,
            "writes_enabled": self._writes_enabled,
            "last_warning": self._last_warning,
            "last_hook_status": self._last_hook_status,
            "last_search_status": self._last_search_status,
            "last_mine_status": self._last_mine_status,
            "last_successful_write_at": self._last_successful_write_at,
            "last_failed_write_at": self._last_failed_write_at,
            "dropped_jobs": self._dropped_jobs,
            "scope_tag": self._scope_tag,
            "agent_context": self._agent_context,
        }

    def _recall_limit(self) -> int:
        return max(1, int(self._config.get("recall_limit") or DEFAULT_RECALL_LIMIT))

    def _search_timeout(self) -> float:
        return float(self._config.get("search_timeout_s") or DEFAULT_SEARCH_TIMEOUT_S)

    def _status_timeout(self) -> float:
        return self._search_timeout()

    def _prefetch_timeout(self) -> float:
        return float(self._config.get("prefetch_timeout_s") or DEFAULT_PREFETCH_TIMEOUT_S)

    def _session_sync_timeout(self) -> float:
        return float(
            self._config.get("session_sync_timeout_s") or DEFAULT_SESSION_SYNC_TIMEOUT_S
        )

    def _enable_wakeup(self) -> bool:
        return as_bool(self._config.get("enable_wakeup"), True)

    def _start_worker(self) -> None:
        self._stop_worker()
        self._queue_maxsize = max(
            1, int(self._config.get("queue_maxsize") or self._DEFAULT_QUEUE_MAXSIZE)
        )
        self._worker_queue = queue.Queue(maxsize=self._queue_maxsize)
        self._accepting_tasks = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="mempalace-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def _stop_worker(self) -> None:
        self._accepting_tasks = False
        if self._worker_queue is None:
            self._worker_thread = None
            return
        self._worker_queue.put(self._worker_sentinel)
        self._worker_queue.join()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=self._QUEUE_JOIN_TIMEOUT)
        self._worker_queue = None
        self._worker_thread = None

    def _worker_loop(self) -> None:
        while True:
            assert self._worker_queue is not None
            job = self._worker_queue.get()
            try:
                if job is self._worker_sentinel:
                    return
                if callable(job):
                    job()
            except Exception as exc:
                self._mark_write_failure(exc)
                logger.warning("MemPalace background task failed: %s", exc, exc_info=True)
            finally:
                self._worker_queue.task_done()

    def _enqueue_task(self, task: Callable[[], None], *, label: str) -> None:
        if (
            not self._accepting_tasks
            or self._worker_queue is None
            or self._worker_thread is None
            or not self._worker_thread.is_alive()
        ):
            self._record_warning(f"Worker unavailable for task '{label}'")
            return
        try:
            self._worker_queue.put(task, timeout=self._QUEUE_PUT_TIMEOUT)
        except queue.Full:
            self._dropped_jobs += 1
            self._record_warning(f"Worker queue full; dropped task '{label}'")

    def _mark_write_success(self, session_id: str) -> None:
        del session_id
        self._last_successful_write_at = datetime.now(timezone.utc).isoformat()

    def _mark_write_failure(self, exc: Exception) -> None:
        self._last_failed_write_at = datetime.now(timezone.utc).isoformat()
        self._record_warning(str(exc))

    def _record_warning(self, warning: str) -> None:
        self._last_warning = warning
        logger.warning("MemPalace warning: %s", warning)

    def _trim_prefetch_cache(self) -> None:
        while len(self._prefetch_cache) > self._MAX_PREFETCH_CACHE_ENTRIES:
            oldest = next(iter(self._prefetch_cache))
            self._prefetch_cache.pop(oldest, None)


def register(ctx):
    ctx.register_memory_provider(MempalaceMemoryProvider())
