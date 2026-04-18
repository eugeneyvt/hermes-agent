# MemPalace Memory Provider

MemPalace is the CLI-first Hermes memory provider.

Hermes shells out to the public MemPalace CLI and hook commands, writes Codex-compatible transcript exports, and lets upstream MemPalace own the session ingest pipeline.

## Runtime behavior

- `initialize()` writes a transcript file for the session and triggers `mempalace hook run --hook session-start --harness codex`.
- `sync_turn()` appends user/assistant turns to the transcript export and asynchronously triggers the upstream `stop` hook cadence.
- When the upstream `stop` hook requests a save checkpoint, Hermes runs `mempalace mine <transcript_export_dir> --mode convos`.
- `on_pre_compress()` triggers the upstream `precompact` hook and then runs a synchronous conversation mine so context is filed before compression.
- `prefetch()` uses `mempalace wake-up` on the first turn when enabled, then `mempalace search` for subsequent recall.
- Tools are exposed as `mempalace_*` names mirroring the upstream MemPalace MCP tools, for example `mempalace_status`, `mempalace_search`, `mempalace_kg_stats`, and `mempalace_kg_timeline`.

## Configuration

`$HERMES_HOME/mempalace.json`

Important keys:

- `command`: command used to run MemPalace, for example `mempalace` or `python3 -m mempalace`
- `repo_path`: optional local MemPalace checkout; when `command` is omitted Hermes tries `.venv/bin/mempalace` there first, then `uv run --project <repo_path> mempalace`
- `palace_path`: optional explicit palace path passed through as `--palace`
- `transcript_export_dir`: where Hermes writes session transcript JSONL files
- `enable_wakeup`: whether to use `mempalace wake-up` for first-turn recall
- `conversation_wing`: base wing name for mined Hermes conversations; Hermes appends profile/user scope suffixes during automatic writes

## Operator workflow

- `hermes mempalace status`
- `hermes mempalace doctor`
- `hermes mempalace search "query"`
- `hermes mempalace wakeup`
- `hermes mempalace flush`

These commands delegate to the upstream MemPalace CLI and surface Hermes transcript-path diagnostics.

## Known limitations

- This provider exposes the public upstream MemPalace MCP tools under `mempalace_*` names such as `mempalace_status`, `mempalace_search`, `mempalace_kg_stats`, `mempalace_kg_timeline`, and `mempalace_list_drawers`.
- Direct write helpers such as manual filing and KG mutation are intentionally not wired through private Python APIs.
- Search results are returned as raw CLI output from upstream MemPalace, not as a Hermes-native parsed hit list.
