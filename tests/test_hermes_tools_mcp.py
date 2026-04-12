import hermes_tools_mcp


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
    monkeypatch.setattr(
        hermes_tools_mcp,
        "get_tool_definitions",
        lambda quiet_mode=True: _tool_defs("read_file", "terminal", "memory"),
    )

    exported = hermes_tools_mcp._exportable_schemas()

    assert [tool["name"] for tool in exported] == ["read_file"]
