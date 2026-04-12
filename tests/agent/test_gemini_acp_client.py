import json
from unittest.mock import MagicMock

from agent.gemini_acp_client import GeminiACPClient, _runtime_settings_path, _tools_mcp_command
import agent.gemini_acp_client as gemini_acp_client


def test_tools_mcp_command_keeps_venv_python(monkeypatch):
    monkeypatch.delenv("HERMES_TOOLS_MCP_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_PYTHON", raising=False)
    monkeypatch.setattr("os.sys.executable", "/tmp/hermes-venv/bin/python")

    assert _tools_mcp_command() == "/tmp/hermes-venv/bin/python"


def test_mcp_servers_payload_uses_persistent_http_server(monkeypatch):
    monkeypatch.setattr(
        "agent.gemini_acp_client._TOOLS_MCP_HTTP_SERVER.ensure_started",
        lambda timeout_seconds: "http://127.0.0.1:8765/mcp",
    )

    client = GeminiACPClient(command="/usr/bin/true", args=["--acp"])
    payload = client._mcp_servers_payload(("read_file", "terminal"))

    assert payload == [
        {
            "name": "hermes-tools",
            "type": "http",
            "url": "http://127.0.0.1:8765/mcp",
            "headers": [],
            "includeTools": ["read_file", "terminal"],
        }
    ]


def test_handle_session_update_fires_stream_and_reasoning_callbacks():
    stream_chunks = []
    reasoning_chunks = []
    client = GeminiACPClient(command="/usr/bin/true", args=["--acp"])
    client.set_stream_handlers(
        stream_delta_callback=stream_chunks.append,
        reasoning_callback=reasoning_chunks.append,
    )

    text_parts = []
    reasoning_parts = []

    assert client._handle_server_message(
        {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"text": "Hello"},
                }
            },
        },
        process=type("P", (), {"stdin": None})(),
        text_parts=text_parts,
        reasoning_parts=reasoning_parts,
        usage_box={},
    )
    assert client._handle_server_message(
        {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"text": "Think"},
                }
            },
        },
        process=type("P", (), {"stdin": None})(),
        text_parts=text_parts,
        reasoning_parts=reasoning_parts,
        usage_box={},
    )

    assert text_parts == ["Hello"]
    assert reasoning_parts == ["Think"]
    assert stream_chunks == ["Hello"]
    assert reasoning_chunks == ["Think"]


def test_handle_tool_event_fires_progress_and_lifecycle_callbacks():
    progress_events = []
    starts = []
    completes = []

    client = GeminiACPClient(command="/usr/bin/true", args=["--acp"])
    client.set_stream_handlers(
        tool_progress_callback=lambda *args, **kwargs: progress_events.append((args, kwargs)),
        tool_start_callback=lambda tool_call_id, function_name, function_args: starts.append(
            (tool_call_id, function_name, function_args)
        ),
        tool_complete_callback=lambda tool_call_id, function_name, function_args, function_result: completes.append(
            (tool_call_id, function_name, function_args, function_result)
        ),
    )

    client._handle_tool_event(
        {
            "event_type": "tool.started",
            "tool_name": "homeassistant",
            "tool_call_id": "mcp-call-1",
            "preview": "turn on kitchen lights",
            "args": {"entity_id": "light.kitchen"},
        }
    )
    client._handle_tool_event(
        {
            "event_type": "tool.completed",
            "tool_name": "homeassistant",
            "tool_call_id": "mcp-call-1",
            "args": {"entity_id": "light.kitchen"},
            "result": '{"success": true}',
            "duration": 0.42,
            "is_error": False,
        }
    )

    assert starts == [("mcp-call-1", "homeassistant", {"entity_id": "light.kitchen"})]
    assert completes == [
        ("mcp-call-1", "homeassistant", {"entity_id": "light.kitchen"}, '{"success": true}')
    ]
    assert progress_events[0] == (
        ("tool.started", "homeassistant", "turn on kitchen lights", {"entity_id": "light.kitchen"}),
        {},
    )
    assert progress_events[1] == (
        ("tool.completed", "homeassistant", None, None),
        {"duration": 0.42, "is_error": False},
    )


def test_parse_tool_event_signal_line_extracts_payload():
    line = gemini_acp_client._TOOL_EVENT_SIGNAL_PREFIX + json.dumps(
        {
            "kind": "tool_event",
            "token": "tok-123",
            "payload": {
                "event_type": "tool.started",
                "tool_name": "terminal",
                "tool_call_id": "mcp-call-1",
                "args": {"command": "pwd"},
            },
        }
    )

    assert gemini_acp_client._parse_tool_event_signal_line(line) == (
        "tok-123",
        {
            "event_type": "tool.started",
            "tool_name": "terminal",
            "tool_call_id": "mcp-call-1",
            "args": {"command": "pwd"},
        },
    )


def test_tools_mcp_stderr_signal_dispatches_to_single_callback(monkeypatch):
    server = gemini_acp_client._PersistentToolsMCPHTTPServer()
    dispatch = MagicMock()
    monkeypatch.setattr(gemini_acp_client._TOOL_EVENT_RELAY, "dispatch_signal", dispatch)

    server._handle_stderr_line(
        gemini_acp_client._TOOL_EVENT_SIGNAL_PREFIX
        + json.dumps(
            {
                "kind": "tool_event",
                "token": "tok-123",
                "payload": {
                    "event_type": "tool.completed",
                    "tool_name": "terminal",
                    "tool_call_id": "mcp-call-1",
                    "result": '{"output":"/tmp"}',
                },
            }
        )
    )

    dispatch.assert_called_once_with(
        "tok-123",
        {
            "event_type": "tool.completed",
            "tool_name": "terminal",
            "tool_call_id": "mcp-call-1",
            "result": '{"output":"/tmp"}',
        },
    )


def test_handle_tool_event_prefers_raw_tool_signal_callback():
    raw_events = []
    client = GeminiACPClient(command="/usr/bin/true", args=["--acp"])
    client.set_stream_handlers(tool_signal_callback=raw_events.append)

    client._handle_tool_event(
        {
            "event_type": "tool.started",
            "tool_name": "terminal",
            "tool_call_id": "mcp-call-1",
            "args": {"command": "pwd"},
        }
    )

    assert raw_events == [
        {
            "event_type": "tool.started",
            "tool_name": "terminal",
            "tool_call_id": "mcp-call-1",
            "args": {"command": "pwd"},
        }
    ]


def test_mcp_servers_payload_includes_tool_event_headers_when_callbacks_enabled(monkeypatch):
    monkeypatch.setattr(
        "agent.gemini_acp_client._TOOLS_MCP_HTTP_SERVER.ensure_started",
        lambda timeout_seconds: "http://127.0.0.1:8765/mcp",
    )
    monkeypatch.setattr(
        "agent.gemini_acp_client._TOOL_EVENT_RELAY.ensure_started",
        lambda: "http://127.0.0.1:9999/",
    )

    client = GeminiACPClient(command="/usr/bin/true", args=["--acp"])
    client.set_stream_handlers(tool_progress_callback=lambda *args, **kwargs: None)
    payload = client._mcp_servers_payload(("read_file",))

    headers = payload[0]["headers"]
    assert {"name": "X-Hermes-Tool-Event-Url", "value": "http://127.0.0.1:9999/"} in headers
    token_headers = [header for header in headers if header["name"] == "X-Hermes-Tool-Event-Token"]
    assert len(token_headers) == 1
    assert token_headers[0]["value"]


def test_runtime_settings_path_honors_builtin_tools_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GEMINI_ACP_BUILTIN_TOOLS", "web_fetch")

    path = _runtime_settings_path()

    assert path is not None
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"tools": {"core": ["web_fetch"]}}
