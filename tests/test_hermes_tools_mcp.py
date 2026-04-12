import pytest

import hermes_tools_mcp
from io import StringIO


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def test_exportable_schemas_uses_default_subset_when_include_env_absent(monkeypatch):
    monkeypatch.delenv("HERMES_TOOLS_MCP_INCLUDE", raising=False)
    monkeypatch.delenv("HERMES_TOOLS_MCP_EXCLUDE", raising=False)
    monkeypatch.delenv("HERMES_TOOLS_MCP_ALLOW_ALL", raising=False)
    monkeypatch.setattr(
        hermes_tools_mcp,
        "get_tool_definitions",
        lambda quiet_mode=True: _tool_defs("read_file", "memory", "delegate", "unknown_tool"),
    )

    exported = hermes_tools_mcp._exportable_schemas()

    assert [tool["name"] for tool in exported] == ["read_file"]


def test_exportable_schemas_respects_explicit_empty_include(monkeypatch):
    monkeypatch.setenv("HERMES_TOOLS_MCP_INCLUDE", "")
    monkeypatch.delenv("HERMES_TOOLS_MCP_EXCLUDE", raising=False)
    monkeypatch.delenv("HERMES_TOOLS_MCP_ALLOW_ALL", raising=False)
    monkeypatch.setattr(
        hermes_tools_mcp,
        "get_tool_definitions",
        lambda quiet_mode=True: _tool_defs("read_file", "terminal"),
    )

    exported = hermes_tools_mcp._exportable_schemas()

    assert exported == []


def test_exportable_schemas_filters_include_and_exclude(monkeypatch):
    monkeypatch.setenv("HERMES_TOOLS_MCP_INCLUDE", "read_file,terminal,memory")
    monkeypatch.setenv("HERMES_TOOLS_MCP_EXCLUDE", "terminal")
    monkeypatch.delenv("HERMES_TOOLS_MCP_ALLOW_ALL", raising=False)
    monkeypatch.setattr(
        hermes_tools_mcp,
        "get_tool_definitions",
        lambda quiet_mode=True: _tool_defs("read_file", "terminal", "memory"),
    )

    exported = hermes_tools_mcp._exportable_schemas()

    assert [tool["name"] for tool in exported] == ["read_file"]


def test_exportable_schemas_allow_all_exports_non_excluded_tools(monkeypatch):
    monkeypatch.delenv("HERMES_TOOLS_MCP_INCLUDE", raising=False)
    monkeypatch.delenv("HERMES_TOOLS_MCP_EXCLUDE", raising=False)
    monkeypatch.setenv("HERMES_TOOLS_MCP_ALLOW_ALL", "1")
    monkeypatch.setattr(
        hermes_tools_mcp,
        "get_tool_definitions",
        lambda quiet_mode=True: _tool_defs("read_file", "terminal", "memory", "delegate"),
    )

    exported = hermes_tools_mcp._exportable_schemas()

    assert [tool["name"] for tool in exported] == ["read_file", "terminal"]


def test_make_handler_emits_tool_events(monkeypatch):
    events = []

    class FakeEmitter:
        @classmethod
        def from_context(cls, ctx):
            return cls()

        def emit_started(self, tool_name, tool_call_id, args):
            events.append(("started", tool_name, tool_call_id, args))

        def emit_completed(self, tool_name, tool_call_id, args, result, duration, is_error):
            events.append(("completed", tool_name, tool_call_id, args, result, is_error))

    monkeypatch.setattr(hermes_tools_mcp, "_ToolEventEmitter", FakeEmitter)
    monkeypatch.setattr(
        hermes_tools_mcp.registry,
        "dispatch",
        lambda tool_name, args, task_id=None: '{"success": true}',
    )

    handler = hermes_tools_mcp._make_handler(
        "homeassistant",
        "homeassistant tool",
        {"type": "object", "properties": {"entity_id": {"type": "string"}}},
    )
    result = handler(entity_id="light.kitchen")

    assert result == '{"success": true}'
    assert events[0][0] == "started"
    assert events[0][1] == "homeassistant"
    assert events[0][3] == {"entity_id": "light.kitchen"}
    assert events[1] == (
        "completed",
        "homeassistant",
        events[0][2],
        {"entity_id": "light.kitchen"},
        '{"success": true}',
        False,
    )


def test_tool_event_emitter_writes_structured_stderr_signal(monkeypatch):
    stderr_buf = StringIO()
    posted = []
    emitter = hermes_tools_mcp._ToolEventEmitter(
        url="http://example.test/events",
        token="tok-123",
    )

    monkeypatch.setattr(hermes_tools_mcp.sys, "stderr", stderr_buf)
    monkeypatch.setattr(
        hermes_tools_mcp._ToolEventEmitter,
        "_post",
        lambda self, payload: posted.append(payload),
    )

    emitter.emit_started("terminal", "mcp-call-1", {"command": "pwd"})

    line = stderr_buf.getvalue().strip()
    assert line.startswith(hermes_tools_mcp._TOOL_EVENT_SIGNAL_PREFIX)
    envelope = hermes_tools_mcp.json.loads(
        line[len(hermes_tools_mcp._TOOL_EVENT_SIGNAL_PREFIX):]
    )
    assert envelope == {
        "kind": "tool_event",
        "token": "tok-123",
        "payload": {
            "event_type": "tool.started",
            "tool_name": "terminal",
            "tool_call_id": "mcp-call-1",
            "preview": "pwd",
            "args": {"command": "pwd"},
        },
    }
    assert posted == [envelope["payload"]]


def test_make_handler_emits_completed_event_on_dispatch_exception(monkeypatch):
    events = []

    class FakeEmitter:
        @classmethod
        def from_context(cls, ctx):
            return cls()

        def emit_started(self, tool_name, tool_call_id, args):
            events.append(("started", tool_name, tool_call_id, args))

        def emit_completed(self, tool_name, tool_call_id, args, result, duration, is_error):
            events.append(("completed", tool_name, tool_call_id, args, result, is_error))

    def _boom(tool_name, args, task_id=None):
        raise RuntimeError("bridge exploded")

    monkeypatch.setattr(hermes_tools_mcp, "_ToolEventEmitter", FakeEmitter)
    monkeypatch.setattr(hermes_tools_mcp.registry, "dispatch", _boom)

    handler = hermes_tools_mcp._make_handler(
        "terminal",
        "terminal tool",
        {"type": "object", "properties": {"command": {"type": "string"}}},
    )

    with pytest.raises(RuntimeError, match="bridge exploded"):
        handler(command="pwd")

    assert events[0][0] == "started"
    assert events[1] == (
        "completed",
        "terminal",
        events[0][2],
        {"command": "pwd"},
        "Error executing tool 'terminal': bridge exploded",
        True,
    )
