# MemPalace Memory Provider

MemPalace is a CLI-first Hermes memory provider.

Hermes writes Codex-compatible transcript exports, uses the upstream CLI/hook surface for session lifecycle ingest and mining, and uses the public MemPalace MCP tools for runtime status/search/KG access.

## Setup flow

Minimal setup for a new user:

1. Install MemPalace in an environment Hermes can reach.
2. Set `memory.provider: mempalace` in Hermes config.
3. If your palace lives outside the MemPalace default location, set `palace_path` in `$HERMES_HOME/mempalace.json`.
4. Start a new Hermes session. If autodetect fails, set `command` explicitly in `$HERMES_HOME/mempalace.json`.

Hermes config:

```yaml
memory:
  provider: mempalace
```

Example minimal config:

```json
{
  "palace_path": "/path/to/palace",
  "command": "mempalace",
  "enable_wakeup": true
}
```

Runtime autodetect order when `command` is blank:

1. `$HERMES_HOME/mempalace.json:repo_path/.venv/bin/mempalace`
2. `uv run --project <repo_path> mempalace`
3. `mempalace` from `PATH`

For the MCP server, Hermes uses the same configuration and tries to launch `mempalace.mcp_server` in the matching Python environment. If the CLI is found but the MCP server cannot be imported, set `command` or `repo_path` explicitly so Hermes can resolve both runtime paths from the same installation.

## Runtime behavior

- `initialize()` writes a transcript file for the session and triggers `mempalace hook run --hook session-start --harness codex`.
- `sync_turn()` appends user/assistant turns to the transcript export and asynchronously triggers the upstream `stop` hook cadence.
- When the upstream `stop` hook requests a save checkpoint, Hermes runs `mempalace mine <transcript_export_dir> --mode convos`.
- `on_pre_compress()` triggers the upstream `precompact` hook and then runs a synchronous conversation mine so context is filed before compression.
- `prefetch()` uses `mempalace wake-up` on the first turn when enabled, then `mempalace search` for subsequent recall.
- Tools are exposed as `mempalace_*` names mirroring the upstream MemPalace MCP tools, for example `mempalace_status`, `mempalace_search`, `mempalace_kg_stats`, and `mempalace_kg_timeline`.
- Runtime tool calls go through Hermes' MCP integration layer, while the CLI path owns hooks, transcript mining, and operator diagnostics.

## Configuration

`$HERMES_HOME/mempalace.json`

Basic keys:

- `palace_path`: optional explicit palace path passed through as `--palace`
- `command`: optional command override used to run MemPalace CLI operations; examples: `mempalace`, `python3 -m mempalace`
- `enable_wakeup`: whether to use `mempalace wake-up` for first-turn recall

Advanced keys:

- `repo_path`: optional local MemPalace checkout; when `command` is omitted Hermes tries `.venv/bin/mempalace` there first, then `uv run --project <repo_path> mempalace`
- `transcript_export_dir`: where Hermes writes session transcript JSONL files
- `recall_limit`: default number of search results requested from MemPalace
- `search_timeout_s`: timeout for search and wake-up CLI commands
- `session_sync_timeout_s`: timeout for synchronous session save and mine operations
- `prefetch_timeout_s`: timeout for background prefetch search operations
- `queue_maxsize`: maximum number of background MemPalace jobs queued by Hermes
- `agent_name`: logical agent/profile name used for transcript metadata
- `scope_by_profile`: whether to isolate memory by Hermes profile name
- `scope_by_user`: whether to isolate memory by gateway user ID when present
- `conversation_wing`: base wing name for mined Hermes conversation transcripts before profile/user scope suffixes are applied
- `flush_wing`: optional CLI flush alias for `conversation_wing`
- `memory_wing`: base wing name for Hermes direct-write durable facts and delegation records
- `memory_room`: room used for built-in `MEMORY.md` fact mirrors
- `user_room`: room used for built-in `USER.md` fact mirrors
- `delegation_room`: room used for parent-side delegation summaries
- `direct_write_max_chars`: maximum characters persisted for direct-write fact and delegation mirrors
- `enabled_tools`: optional extra MemPalace tools to expose beyond the provider defaults
- `disabled_tools`: optional MemPalace tools to hide even if they are enabled by default

Example full config:

```json
{
  "command": "mempalace",
  "repo_path": "",
  "palace_path": "/path/to/palace",
  "transcript_export_dir": "",
  "enable_wakeup": true,
  "recall_limit": 5,
  "search_timeout_s": 8.0,
  "session_sync_timeout_s": 30.0,
  "prefetch_timeout_s": 8.0,
  "queue_maxsize": 32,
  "agent_name": "hermes",
  "scope_by_profile": true,
  "scope_by_user": true,
  "memory_wing": "wing_hermes_memory",
  "memory_room": "memory",
  "user_room": "user",
  "delegation_room": "delegations",
  "direct_write_max_chars": 4000,
  "enabled_tools": [],
  "disabled_tools": []
}
```

Tool policy:

- If `enabled_tools` and `disabled_tools` are omitted, the provider exposes its default curated MemPalace tool set.
- `enabled_tools` adds extra public tools on top of the defaults.
- `disabled_tools` hides tools from the default set.

Default curated tool set:

- `mempalace_status`
- `mempalace_list_wings`
- `mempalace_list_rooms`
- `mempalace_get_taxonomy`
- `mempalace_kg_query`
- `mempalace_kg_add`
- `mempalace_kg_invalidate`
- `mempalace_kg_timeline`
- `mempalace_kg_stats`
- `mempalace_traverse`
- `mempalace_find_tunnels`
- `mempalace_graph_stats`
- `mempalace_create_tunnel`
- `mempalace_list_tunnels`
- `mempalace_delete_tunnel`
- `mempalace_follow_tunnels`
- `mempalace_search`
- `mempalace_check_duplicate`
- `mempalace_add_drawer`
- `mempalace_delete_drawer`
- `mempalace_get_drawer`
- `mempalace_list_drawers`
- `mempalace_update_drawer`

## Operator workflow

These commands delegate to the upstream MemPalace CLI and surface hook/transcript-path diagnostics.

- `hermes mempalace status`
  Runs the upstream `mempalace status` command using the provider's configured runtime and `palace_path`.

- `hermes mempalace doctor`
  Runs basic health checks for the configured MemPalace CLI path, hook entrypoints, and Hermes transcript export path.

- `hermes mempalace search "query"`
  Runs a manual MemPalace search outside the normal agent loop. Useful for checking whether the palace contains the memory you expect.

- `hermes mempalace wakeup`
  Shows the upstream `mempalace wake-up` output for the current provider configuration. Useful for validating first-turn recall behavior.

- `hermes mempalace flush`
  Runs a conversation mine over Hermes transcript exports so pending session transcripts are filed into the palace immediately.

## Known limitations

- This provider exposes a curated subset of the upstream MemPalace MCP tools under `mempalace_*` names by default. You can widen or narrow that set with `enabled_tools` and `disabled_tools` in `mempalace.json`.
- Direct write helpers such as manual filing and KG mutation are intentionally not wired through private Python APIs.
- Search results are returned as raw CLI output from upstream MemPalace, not as a Hermes-native parsed hit list.
