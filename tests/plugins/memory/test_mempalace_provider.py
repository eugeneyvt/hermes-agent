import json
import subprocess
from pathlib import Path

import pytest

from plugins.memory.mempalace import cli as mempalace_cli
from plugins.memory.mempalace import MempalaceMemoryProvider
from plugins.memory.mempalace.config import load_provider_config, save_provider_config


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=["mempalace"], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.fixture
def provider(tmp_path, monkeypatch):
    save_provider_config(
        {
            "command": "python3 -m mempalace",
            "palace_path": str(tmp_path / "palace"),
            "enable_wakeup": True,
            "recall_limit": 4,
            "agent_name": "hermes",
            "queue_maxsize": 4,
        },
        str(tmp_path),
    )

    cli_calls = []
    mcp_calls = []
    connection_state = {"connected": False}
    mcp_schemas = [
        {
            "name": "mcp_mempalace_mempalace_status",
            "description": "MemPalace status",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "mcp_mempalace_mempalace_search",
            "description": "MemPalace search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    ]

    def fake_run(cmd, **kwargs):
        cli_calls.append((cmd, kwargs))
        if "hook" in cmd:
            return _completed(stdout='{"decision":"block","reason":"save"}\n')
        if "mine" in cmd:
            return _completed(stdout="mine ok\n")
        if "wake-up" in cmd:
            return _completed(stdout="wake up text\n")
        if cmd[-1] == "status":
            return _completed(stdout="status ok\n")
        return _completed(stdout="")

    def fake_register(servers):
        assert "mempalace" in servers
        connection_state["connected"] = True
        return ["mcp_mempalace_mempalace_status", "mcp_mempalace_mempalace_search"]

    def fake_is_connected(server_name):
        assert server_name == "mempalace"
        return connection_state["connected"]

    def fake_get_schemas(server_name):
        assert server_name == "mempalace"
        return list(mcp_schemas) if connection_state["connected"] else []

    def fake_call(server_name, tool_name, arguments=None, timeout=None):
        assert server_name == "mempalace"
        mcp_calls.append((tool_name, arguments or {}, timeout))
        if tool_name == "mempalace_status":
            return {"result": {"total_drawers": 5, "palace_path": str(tmp_path / "palace")}}
        if tool_name == "mempalace_search":
            return {"result": {"hits": [{"text": "search results"}], "query": arguments["query"]}}
        if tool_name == "mempalace_add_drawer":
            return {
                "result": {
                    "success": True,
                    "drawer_id": "drawer-1",
                    "wing": arguments["wing"],
                    "room": arguments["room"],
                }
            }
        if tool_name == "mempalace_reconnect":
            return {"result": {"ok": True}}
        return {"error": f"unknown tool {tool_name}"}

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("tools.mcp_tool.register_mcp_servers", fake_register)
    monkeypatch.setattr("tools.mcp_tool.is_mcp_server_connected", fake_is_connected)
    monkeypatch.setattr("tools.mcp_tool.get_mcp_tool_schemas", fake_get_schemas)
    monkeypatch.setattr("tools.mcp_tool.call_mcp_tool", fake_call)

    p = MempalaceMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
    yield p, cli_calls, mcp_calls
    p.shutdown()


def test_is_available_uses_mcp_status(tmp_path, monkeypatch):
    save_provider_config({"command": "python3 -m mempalace"}, str(tmp_path))
    monkeypatch.setattr("tools.mcp_tool.register_mcp_servers", lambda servers: ["mcp_mempalace_mempalace_status"])
    monkeypatch.setattr("tools.mcp_tool.is_mcp_server_connected", lambda name: True)
    monkeypatch.setattr(
        "tools.mcp_tool.get_mcp_tool_schemas",
        lambda name: [
            {
                "name": "mcp_mempalace_mempalace_status",
                "description": "MemPalace status",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    monkeypatch.setattr(
        "tools.mcp_tool.call_mcp_tool",
        lambda server_name, tool_name, arguments=None, timeout=None: {"result": {"total_drawers": 1}},
    )

    provider = MempalaceMemoryProvider()
    provider._hermes_home = str(tmp_path)
    assert provider.is_available() is True


def test_is_available_accepts_degraded_palace_status(tmp_path, monkeypatch):
    save_provider_config({"command": "python3 -m mempalace"}, str(tmp_path))
    monkeypatch.setattr("tools.mcp_tool.register_mcp_servers", lambda servers: ["mcp_mempalace_mempalace_status"])
    monkeypatch.setattr("tools.mcp_tool.is_mcp_server_connected", lambda name: True)
    monkeypatch.setattr(
        "tools.mcp_tool.get_mcp_tool_schemas",
        lambda name: [
            {
                "name": "mcp_mempalace_mempalace_status",
                "description": "MemPalace status",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    monkeypatch.setattr(
        "tools.mcp_tool.call_mcp_tool",
        lambda server_name, tool_name, arguments=None, timeout=None: {
            "result": {"error": "No palace found", "hint": "Run: mempalace init <dir> && mempalace mine <dir>"}
        },
    )

    provider = MempalaceMemoryProvider()
    provider._hermes_home = str(tmp_path)
    assert provider.is_available() is True
    assert provider._palace_healthy is False
    assert "No palace found" in provider._last_warning


def test_initialize_creates_transcript_and_session_start_hook(provider):
    p, cli_calls, _ = provider
    transcript = Path(p._transcript_path)
    assert transcript.exists()
    first_line = json.loads(transcript.read_text(encoding="utf-8").splitlines()[0])
    assert first_line["type"] == "session_meta"
    assert any("hook" in cmd for cmd, _ in cli_calls)


def test_get_tool_schemas_comes_from_connected_mcp(provider):
    p, _, _ = provider
    schemas = p.get_tool_schemas()
    assert {schema["name"] for schema in schemas} == {
        "mempalace_status",
        "mempalace_search",
    }


def test_handle_tool_call_proxies_connected_mcp_tool(provider):
    p, _, mcp_calls = provider
    raw = p.handle_tool_call("mempalace_status", {})
    payload = json.loads(raw)
    assert payload["result"]["total_drawers"] == 5
    assert payload["result"]["provider_runtime"]["mcp_connected"] is True
    assert mcp_calls[0][0] == "mempalace_status"


def test_handle_tool_call_accepts_legacy_prefixed_name(provider):
    p, _, mcp_calls = provider
    raw = p.handle_tool_call("mcp_mempalace_mempalace_status", {})
    payload = json.loads(raw)
    assert payload["result"]["total_drawers"] == 5
    assert mcp_calls[0][0] == "mempalace_status"


def test_sync_turn_appends_codex_events_and_triggers_mine(provider):
    p, cli_calls, mcp_calls = provider

    p.sync_turn("hello", "world")
    p.shutdown()

    lines = [
        json.loads(line)
        for line in Path(p._transcript_path).read_text(encoding="utf-8").splitlines()
    ]
    assert lines[1]["payload"]["type"] == "user_message"
    assert lines[2]["payload"]["type"] == "agent_message"
    assert any("mine" in cmd for cmd, _ in cli_calls)
    assert any(name == "mempalace_reconnect" for name, _, _ in mcp_calls) is False


def test_sync_turn_uses_configured_conversation_wing(tmp_path, monkeypatch):
    save_provider_config(
        {
            "command": "python3 -m mempalace",
            "palace_path": str(tmp_path / "palace"),
            "conversation_wing": "wing_custom_sessions",
            "agent_name": "hermes",
        },
        str(tmp_path),
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if "hook" in cmd:
            return _completed(stdout='{"decision":"block","reason":"save"}\n')
        if "mine" in cmd:
            return _completed(stdout="mine ok\n")
        return _completed(stdout="ok\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("tools.mcp_tool.register_mcp_servers", lambda servers: ["mcp_mempalace_mempalace_status"])
    monkeypatch.setattr("tools.mcp_tool.is_mcp_server_connected", lambda name: True)
    monkeypatch.setattr(
        "tools.mcp_tool.get_mcp_tool_schemas",
        lambda name: [
            {
                "name": "mcp_mempalace_mempalace_status",
                "description": "MemPalace status",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    monkeypatch.setattr(
        "tools.mcp_tool.call_mcp_tool",
        lambda server_name, tool_name, arguments=None, timeout=None: {"result": {"total_drawers": 1}},
    )

    provider = MempalaceMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
    provider.sync_turn("hello", "world")
    provider.shutdown()

    mine_calls = [cmd for cmd, _ in calls if "mine" in cmd]
    assert mine_calls
    assert "wing_custom_sessions__profile_hermes" in mine_calls[-1]


def test_prefetch_uses_wakeup_on_first_turn(provider):
    p, cli_calls, mcp_calls = provider
    p.on_turn_start(1, "hi")
    before_mcp_calls = len(mcp_calls)

    result = p.prefetch("auth decisions")

    assert "wake up text" in result
    wakeup_calls = [cmd for cmd, _ in cli_calls if "wake-up" in cmd]
    assert wakeup_calls
    assert "--wing" in wakeup_calls[-1]
    assert "wing_hermes_sessions__profile_hermes" in wakeup_calls[-1]
    assert all(name != "mempalace_search" for name, _, _ in mcp_calls[before_mcp_calls:])


def test_queue_prefetch_primes_cache_from_mcp_search(provider):
    p, _, _ = provider

    p.queue_prefetch("auth decisions", session_id="session-1")
    p.shutdown()

    cached = p.prefetch("auth decisions", session_id="session-1")
    assert '"query": "auth decisions"' in cached


def test_on_pre_compress_reconnects_mcp_after_mine(provider):
    p, cli_calls, mcp_calls = provider

    p.on_pre_compress([])

    assert any("mine" in cmd for cmd, _ in cli_calls)
    assert any(name == "mempalace_reconnect" for name, _, _ in mcp_calls)


def test_on_memory_write_mirrors_to_direct_write_room(provider):
    p, _, mcp_calls = provider

    p.on_memory_write("add", "memory", "Jordan likes concise docs")
    p.shutdown()

    add_calls = [call for call in mcp_calls if call[0] == "mempalace_add_drawer"]
    assert add_calls
    _, args, _ = add_calls[-1]
    assert args["wing"] == "wing_hermes_memory__profile_hermes"
    assert args["room"] == "memory"
    payload = json.loads(args["content"])
    assert payload["kind"] == "hermes_memory"
    assert payload["target"] == "memory"
    assert payload["content"] == "Jordan likes concise docs"


def test_on_memory_write_remove_is_warning_noop(provider):
    p, _, mcp_calls = provider

    p.on_memory_write("remove", "memory", "Jordan likes concise docs")

    assert "does not mirror memory action 'remove'" in p._last_warning
    assert not any(name == "mempalace_add_drawer" for name, _, _ in mcp_calls)


def test_on_delegation_mirrors_summary_to_direct_write_room(provider):
    p, _, mcp_calls = provider

    p.on_delegation("Find the auth bug", "Root cause is stale token cache", child_session_id="child-1")
    p.shutdown()

    add_calls = [call for call in mcp_calls if call[0] == "mempalace_add_drawer"]
    assert add_calls
    _, args, _ = add_calls[-1]
    assert args["room"] == "delegations"
    payload = json.loads(args["content"])
    assert payload["kind"] == "hermes_delegation"
    assert payload["child_session_id"] == "child-1"
    assert payload["task"] == "Find the auth bug"
    assert payload["result"] == "Root cause is stale token cache"


def test_non_primary_context_skips_memory_write_and_delegation(tmp_path, monkeypatch):
    save_provider_config({"command": "python3 -m mempalace", "agent_name": "hermes"}, str(tmp_path))

    mcp_calls = []

    monkeypatch.setattr("tools.mcp_tool.register_mcp_servers", lambda servers: ["mcp_mempalace_mempalace_status"])
    monkeypatch.setattr("tools.mcp_tool.is_mcp_server_connected", lambda name: True)
    monkeypatch.setattr(
        "tools.mcp_tool.get_mcp_tool_schemas",
        lambda name: [{"name": "mcp_mempalace_mempalace_status", "description": "", "parameters": {"type": "object", "properties": {}}}],
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(stdout="ok\n"))

    def fake_call(server_name, tool_name, arguments=None, timeout=None):
        mcp_calls.append((tool_name, arguments or {}, timeout))
        return {"result": {"total_drawers": 1}}

    monkeypatch.setattr("tools.mcp_tool.call_mcp_tool", fake_call)

    provider = MempalaceMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_context="subagent")
    try:
        provider.on_memory_write("add", "memory", "x")
        provider.on_delegation("task", "result", child_session_id="child")
    finally:
        provider.shutdown()

    assert not any(name == "mempalace_add_drawer" for name, _, _ in mcp_calls)


def test_save_config_is_atomic(tmp_path):
    save_provider_config({"command": "mempalace"}, str(tmp_path))
    loaded = load_provider_config(str(tmp_path))
    assert loaded["command"] == "mempalace"


def test_save_config_writes_mcp_server_entry(tmp_path, monkeypatch):
    saved = {}

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {"provider": "mempalace"}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda config: saved.update(config))

    provider = MempalaceMemoryProvider()
    provider.save_config(
        {
            "repo_path": str(tmp_path / "mempalace"),
            "palace_path": str(tmp_path / "palace"),
            "agent_name": "hermes",
        },
        str(tmp_path),
    )

    server = saved["mcp_servers"]["mempalace"]
    assert server["enabled"] is True
    assert "--palace" in server["args"]


def test_save_config_preserves_existing_upstream_keys(tmp_path, monkeypatch):
    save_provider_config(
        {
            "additional_palaces": {"refs": "/tmp/refs"},
            "max_distance": 0.85,
            "memory_room": "memory-notes",
        },
        str(tmp_path),
    )

    saved = {}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda config: saved.update(config))

    provider = MempalaceMemoryProvider()
    provider.save_config({"agent_name": "hermes"}, str(tmp_path))

    loaded = load_provider_config(str(tmp_path))
    assert loaded["additional_palaces"] == {"refs": "/tmp/refs"}
    assert loaded["max_distance"] == 0.85
    assert loaded["memory_room"] == "memory-notes"


def test_repo_path_venv_python_is_preferred_for_mcp(tmp_path):
    repo_path = tmp_path / "mempalace"
    python_path = repo_path / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")

    provider = MempalaceMemoryProvider()
    command, args = provider._resolve_mcp_command(
        {
            "repo_path": str(repo_path),
            "palace_path": str(tmp_path / "palace"),
        }
    )

    assert command == str(python_path)
    assert args[:2] == ["-m", "mempalace.mcp_server"]
    assert args[-2:] == ["--palace", str(tmp_path / "palace")]


def test_explicit_mempalace_script_uses_matching_python_for_mcp(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    python_path = bin_dir / "python"
    python_path.write_text("", encoding="utf-8")
    script_path = bin_dir / "mempalace"
    script_path.write_text(f"#!{python_path}\n", encoding="utf-8")

    provider = MempalaceMemoryProvider()
    command, args = provider._resolve_mcp_command(
        {
            "command": str(script_path),
            "palace_path": str(tmp_path / "palace"),
        }
    )

    assert command == str(python_path)
    assert args[:2] == ["-m", "mempalace.mcp_server"]
    assert args[-2:] == ["--palace", str(tmp_path / "palace")]


def test_initialize_keeps_empty_user_id_out_of_scope(tmp_path, monkeypatch):
    save_provider_config({"command": "python3 -m mempalace", "agent_name": "hermes"}, str(tmp_path))

    monkeypatch.setattr("tools.mcp_tool.register_mcp_servers", lambda servers: ["mcp_mempalace_mempalace_status"])
    monkeypatch.setattr("tools.mcp_tool.is_mcp_server_connected", lambda name: True)
    monkeypatch.setattr(
        "tools.mcp_tool.get_mcp_tool_schemas",
        lambda name: [{"name": "mcp_mempalace_mempalace_status", "description": "", "parameters": {"type": "object", "properties": {}}}],
    )
    monkeypatch.setattr(
        "tools.mcp_tool.call_mcp_tool",
        lambda server_name, tool_name, arguments=None, timeout=None: {"result": {"total_drawers": 1}},
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(stdout="ok\n"))

    provider = MempalaceMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
    try:
        assert provider._scope_tag == "profile_hermes"
        assert provider._conversation_wing() == "wing_hermes_sessions__profile_hermes"
    finally:
        provider.shutdown()


def test_cli_flush_uses_scoped_conversation_wing(tmp_path, monkeypatch, capsys):
    save_provider_config(
        {
            "command": "python3 -m mempalace",
            "palace_path": str(tmp_path / "palace"),
            "conversation_wing": "wing_custom_sessions",
            "agent_name": "hermes",
            "scope_by_profile": True,
            "scope_by_user": True,
        },
        str(tmp_path),
    )

    monkeypatch.setattr(mempalace_cli, "_load_cfg", lambda: load_provider_config(str(tmp_path)))

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _completed(stdout="mine ok\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    mempalace_cli.cmd_flush(None)
    capsys.readouterr()

    mine_calls = [cmd for cmd, _ in calls if "mine" in cmd]
    assert mine_calls
    assert "wing_custom_sessions__profile_hermes" in mine_calls[-1]


def test_cli_wakeup_uses_scoped_conversation_wing_by_default(tmp_path, monkeypatch, capsys):
    save_provider_config(
        {
            "command": "python3 -m mempalace",
            "palace_path": str(tmp_path / "palace"),
            "conversation_wing": "wing_custom_sessions",
            "agent_name": "hermes",
            "scope_by_profile": True,
            "scope_by_user": True,
        },
        str(tmp_path),
    )

    monkeypatch.setattr(mempalace_cli, "_load_cfg", lambda: load_provider_config(str(tmp_path)))

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _completed(stdout="wake up text\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = type("Args", (), {"wing": None})()
    mempalace_cli.cmd_wakeup(args)
    capsys.readouterr()

    wakeup_calls = [cmd for cmd, _ in calls if "wake-up" in cmd]
    assert wakeup_calls
    assert "--wing" in wakeup_calls[-1]
    assert "wing_custom_sessions__profile_hermes" in wakeup_calls[-1]
