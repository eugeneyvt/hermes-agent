from __future__ import annotations

import json
import logging
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .bridge import MCPBridgeMixin
from .config import load_provider_config
from .runtime import RuntimeMixin
from .session_io import SessionIOMixin

logger = logging.getLogger(__name__)


class MempalaceMemoryProvider(RuntimeMixin, MCPBridgeMixin, SessionIOMixin, MemoryProvider):
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
    _DEFAULT_VISIBLE_TOOLS = frozenset(
        {
            "mempalace_status",
            "mempalace_list_wings",
            "mempalace_list_rooms",
            "mempalace_get_taxonomy",
            "mempalace_kg_query",
            "mempalace_kg_add",
            "mempalace_kg_invalidate",
            "mempalace_kg_timeline",
            "mempalace_kg_stats",
            "mempalace_traverse",
            "mempalace_find_tunnels",
            "mempalace_graph_stats",
            "mempalace_create_tunnel",
            "mempalace_list_tunnels",
            "mempalace_delete_tunnel",
            "mempalace_follow_tunnels",
            "mempalace_search",
            "mempalace_check_duplicate",
            "mempalace_add_drawer",
            "mempalace_delete_drawer",
            "mempalace_get_drawer",
            "mempalace_list_drawers",
            "mempalace_update_drawer",
        }
    )

    def __init__(self) -> None:
        self._hermes_home = os.path.expanduser("~/.hermes")
        self._config: dict[str, Any] = {}
        self._session_id = ""
        self._agent_name = "hermes"
        self._agent_identity = "hermes"
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
        self._visible_tool_names: set[str] = set(self._DEFAULT_VISIBLE_TOOLS)
        self._mcp_tool_schemas: list[dict[str, Any]] = []
        self._last_warning = ""
        self._last_hook_status = "idle"
        self._last_search_status = "idle"
        self._last_mine_status = "idle"
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
        raw_user_id = str(kwargs.get("user_id") or "").strip()
        self._user_id = self._normalize_name(raw_user_id) if raw_user_id else ""
        self._agent_name = self._normalize_name(
            self._config.get("agent_name") or self._agent_identity or "hermes"
        )
        self._scope_tag = self._build_scope_tag()
        self._turn_count = 0
        self._prefetch_cache = {}
        self._session_message_fingerprints = set()
        self._last_warning = ""
        self._last_hook_status = "idle"
        self._last_search_status = "idle"
        self._last_mine_status = "idle"
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
                return cached

        try:
            if self._enable_wakeup() and self._turn_count <= 1:
                return self._run_wakeup()

            return self._run_search(query, limit=self._recall_limit())
        except Exception as exc:
            self._record_warning(f"Prefetch failed: {exc}")
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        clean_query = (query or "").strip()
        if not clean_query:
            return
        cache_key = self._prefetch_cache_key(clean_query, session_id=session_id)
        with self._prefetch_lock:
            if cache_key in self._prefetch_cache:
                return

        def _work() -> None:
            try:
                block = self._run_search(clean_query, limit=self._recall_limit())
            except Exception as exc:
                self._record_warning(f"Background prefetch failed: {exc}")
                return
            if not block:
                return
            with self._prefetch_lock:
                self._prefetch_cache[cache_key] = block
                self._trim_prefetch_cache()

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
        if actual_name not in self._visible_tool_names:
            return tool_error(f"MemPalace tool is disabled by provider config: {actual_name}")
        try:
            self._ensure_mcp_connected()
            result = self._call_mcp_tool(actual_name, args)
            if actual_name == self._MCP_STATUS_TOOL:
                result = self._enrich_status_result(result)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            logger.warning("MemPalace MCP tool failed: %s", exc, exc_info=True)
            return tool_error(f"MemPalace MCP tool failed: {exc}")


def register(ctx):
    ctx.register_memory_provider(MempalaceMemoryProvider())
