from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

from .config import (
    DEFAULT_PREFETCH_TIMEOUT_S,
    DEFAULT_RECALL_LIMIT,
    DEFAULT_SEARCH_TIMEOUT_S,
    DEFAULT_SESSION_SYNC_TIMEOUT_S,
    as_bool,
)

logger = logging.getLogger(__name__)


class SessionIOMixin:
    def _session_meta_entry(self) -> dict[str, Any]:
        return {
            "type": "session_meta",
            "payload": {
                "session_id": self._session_id,
                "agent_identity": self._agent_identity,
                "scope_tag": self._scope_tag,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }

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

    def _run_wakeup(self) -> str:
        result = self._run_cli(
            ["wake-up", "--wing", self._conversation_wing()],
            timeout=self._search_timeout(),
            check=True,
        )
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
        args = [
            "mine",
            self._snapshot_export_dir(),
            "--mode",
            "convos",
            "--wing",
            self._conversation_wing(),
            "--agent",
            self._agent_name,
        ]
        self._run_cli(args, timeout=self._session_sync_timeout(), check=True)
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
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text(
            json.dumps(self._session_meta_entry(), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _snapshot_export_dir(self) -> str:
        return str(Path(self._transcript_export_dir) / "snapshots")

    def _live_transcript_has_events(self) -> bool:
        if not self._transcript_path:
            return False
        path = Path(self._transcript_path)
        if not path.is_file():
            return False
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("type") == "event_msg":
                        return True
        except OSError:
            return False
        return False

    def _snapshot_transcript(self, *, reason: str) -> str:
        if not self._transcript_path:
            return ""
        source = Path(self._transcript_path)
        if not source.is_file() or not self._live_transcript_has_events():
            return ""
        snapshot_dir = Path(self._snapshot_export_dir())
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        snapshot_name = f"{self._session_id}__{reason}__{stamp}.jsonl"
        target = snapshot_dir / snapshot_name
        shutil.copy2(source, target)
        return str(target)

    def _reset_live_transcript(self) -> None:
        if not self._transcript_path:
            return
        transcript = Path(self._transcript_path)
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text(
            json.dumps(self._session_meta_entry(), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _snapshot_and_mine(self, *, reason: str, rotate_live: bool) -> bool:
        snapshot_path = self._snapshot_transcript(reason=reason)
        if not snapshot_path:
            self._last_mine_status = f"skip_{reason}_no_events"
            return False
        self._run_mine(background=False, reason=reason)
        if rotate_live:
            self._reset_live_transcript()
        return True

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
