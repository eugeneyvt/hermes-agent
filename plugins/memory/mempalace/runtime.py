from __future__ import annotations

import logging
import os
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class RuntimeMixin:
    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "palace_path",
                "description": "Path to the MemPalace palace directory (leave blank to use the upstream default)",
                "required": False,
                "default": os.path.expanduser("~/.mempalace/palace"),
            },
            {
                "key": "command",
                "description": "Optional MemPalace command override. Leave blank to auto-detect from PATH / repo / active Python environment",
                "required": False,
                "default": "",
            },
            {
                "key": "enable_wakeup",
                "description": "Whether to use CLI 'mempalace wake-up' for first-turn context injection",
                "required": False,
                "default": True,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        existing = load_provider_config(hermes_home)

        def _pick(key: str, default: Any = "") -> Any:
            if key in values:
                return values.get(key)
            if key in existing:
                return existing.get(key)
            return default

        sanitized = {
            "repo_path": str(_pick("repo_path", "") or "").strip(),
            "command": str(_pick("command", "") or "").strip(),
            "palace_path": str(_pick("palace_path", "") or "").strip(),
            "transcript_export_dir": str(_pick("transcript_export_dir", "") or "").strip(),
            "enable_wakeup": as_bool(_pick("enable_wakeup", True), True),
            "recall_limit": int(_pick("recall_limit", DEFAULT_RECALL_LIMIT) or DEFAULT_RECALL_LIMIT),
            "search_timeout_s": float(
                _pick("search_timeout_s", DEFAULT_SEARCH_TIMEOUT_S) or DEFAULT_SEARCH_TIMEOUT_S
            ),
            "session_sync_timeout_s": float(
                _pick("session_sync_timeout_s", DEFAULT_SESSION_SYNC_TIMEOUT_S)
                or DEFAULT_SESSION_SYNC_TIMEOUT_S
            ),
            "prefetch_timeout_s": float(
                _pick("prefetch_timeout_s", DEFAULT_PREFETCH_TIMEOUT_S) or DEFAULT_PREFETCH_TIMEOUT_S
            ),
            "queue_maxsize": max(
                1,
                int(_pick("queue_maxsize", self._DEFAULT_QUEUE_MAXSIZE) or self._DEFAULT_QUEUE_MAXSIZE),
            ),
            "agent_name": self._normalize_name(_pick("agent_name", "hermes") or "hermes"),
            "scope_by_profile": as_bool(_pick("scope_by_profile", True), True),
            "scope_by_user": as_bool(_pick("scope_by_user", True), True),
            "memory_wing": self._normalize_name(
                _pick("memory_wing", "wing_hermes_memory") or "wing_hermes_memory"
            ),
            "memory_room": self._normalize_name(_pick("memory_room", "memory") or "memory"),
            "user_room": self._normalize_name(_pick("user_room", "user") or "user"),
            "delegation_room": self._normalize_name(
                _pick("delegation_room", "delegations") or "delegations"
            ),
            "direct_write_max_chars": max(
                256,
                int(
                    _pick("direct_write_max_chars", self._DEFAULT_DIRECT_WRITE_MAX_CHARS)
                    or self._DEFAULT_DIRECT_WRITE_MAX_CHARS
                ),
            ),
            "enabled_tools": self._normalize_tool_list(_pick("enabled_tools", [])),
            "disabled_tools": self._normalize_tool_list(_pick("disabled_tools", [])),
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
            command_parts, command_cwd = self._autodetect_default_command(repo_path)

        if command_parts and command_parts[0] == "mempalace":
            resolved = shutil.which(command_parts[0]) or command_parts[0]
            command_parts[0] = resolved

        self._command = command_parts
        self._resolved_command = " ".join(shlex.quote(part) for part in command_parts)
        self._command_cwd = command_cwd

    def _autodetect_default_command(self, repo_path: str) -> tuple[list[str], str]:
        if repo_path:
            venv_cmd = Path(repo_path) / ".venv" / "bin" / "mempalace"
            if venv_cmd.exists():
                return [str(venv_cmd)], ""
            if shutil.which("uv"):
                return ["uv", "run", "--project", repo_path, "mempalace"], ""

        resolved = shutil.which("mempalace")
        if resolved:
            return [resolved], ""

        fallback = default_cli_command()
        return (shlex.split(fallback) if fallback else []), ""

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
            "timeout": max(
                30,
                int(float(cfg.get("session_sync_timeout_s") or DEFAULT_SESSION_SYNC_TIMEOUT_S)),
            ),
            "connect_timeout": max(
                30, int(float(cfg.get("search_timeout_s") or DEFAULT_SEARCH_TIMEOUT_S))
            ),
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
                mempalace_cmd = shutil.which("mempalace")
                inferred_python = self._infer_python_from_command(mempalace_cmd) if mempalace_cmd else ""
                python_cmd = inferred_python or shutil.which("python3") or shutil.which("python") or "python3"
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

