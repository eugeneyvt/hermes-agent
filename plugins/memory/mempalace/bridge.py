from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MCPBridgeMixin:
    @classmethod
    def _normalize_tool_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            items = [part.strip() for part in value.split(",")]
        elif isinstance(value, (list, tuple, set)):
            items = [str(part).strip() for part in value]
        else:
            return []

        normalized: list[str] = []
        for item in items:
            actual_name = cls._resolve_tool_name(item) or item
            if actual_name.startswith(cls._PUBLIC_PREFIX) and actual_name not in normalized:
                normalized.append(actual_name)
        return normalized

    def _configured_visible_tools(self) -> set[str]:
        visible = set(self._DEFAULT_VISIBLE_TOOLS)
        visible.update(self._normalize_tool_list(self._config.get("enabled_tools") or []))
        visible.difference_update(self._normalize_tool_list(self._config.get("disabled_tools") or []))
        return visible

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
        public_schemas = [
            self._public_tool_schema(schema)
            for schema in schemas
            if schema.get("name", "").startswith(self._MCP_PREFIX)
        ]
        visible = self._configured_visible_tools()
        self._visible_tool_names = {
            schema.get("name", "")
            for schema in public_schemas
            if schema.get("name", "") in visible
        }
        self._mcp_tool_schemas = [
            schema for schema in public_schemas if schema.get("name", "") in self._visible_tool_names
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

