"""CLI commands for the active MemPalace plugin."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from hermes_constants import get_hermes_home

from .config import default_cli_command, load_provider_config


def _load_cfg() -> dict:
    return load_provider_config(str(get_hermes_home()))


def _resolve_command(cfg: dict) -> tuple[list[str], str]:
    command = str(cfg.get("command") or "").strip()
    repo_path = str(cfg.get("repo_path") or "").strip()
    repo_path = os.path.abspath(os.path.expanduser(repo_path)) if repo_path else ""
    cwd = ""

    if command:
        parts = shlex.split(command)
    else:
        venv_cmd = Path(repo_path) / ".venv" / "bin" / "mempalace" if repo_path else None
        if venv_cmd and venv_cmd.exists():
            parts = [str(venv_cmd)]
        elif repo_path and shutil.which("uv"):
            parts = ["uv", "run", "--project", repo_path, "mempalace"]
        else:
            fallback = default_cli_command()
            parts = shlex.split(fallback) if fallback else []
            if repo_path and parts[:3] == ["python3", "-m", "mempalace"]:
                cwd = repo_path

    if parts and parts[0] == "mempalace":
        parts[0] = shutil.which(parts[0]) or parts[0]
    return parts, cwd


def _base_command(cfg: dict) -> tuple[list[str], str]:
    parts, cwd = _resolve_command(cfg)
    if not parts:
        raise RuntimeError("MemPalace command is not configured")
    palace_path = str(cfg.get("palace_path") or "").strip()
    if palace_path:
        parts = parts + ["--palace", palace_path]
    return parts, cwd


def _run_cli(cfg: dict, args: list[str], *, stdin_json: dict | None = None, timeout: float = 30.0):
    base, cwd = _base_command(cfg)
    command = base + args
    result = subprocess.run(
        command,
        cwd=cwd or None,
        input=json.dumps(stdin_json) if stdin_json is not None else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return command, result


def _transcript_dir(cfg: dict) -> str:
    configured = str(cfg.get("transcript_export_dir") or "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return str(Path(get_hermes_home()) / "mempalace_transcripts")


def _normalize_name(value: str) -> str:
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


def _conversation_wing(cfg: dict, *, user_id: str = "") -> str:
    base = _normalize_name(str(cfg.get("flush_wing") or cfg.get("conversation_wing") or "wing_hermes_sessions"))
    parts = []
    if str(cfg.get("scope_by_profile", True)).lower() not in ("0", "false", "no", "off"):
        parts.append(f"profile_{_normalize_name(str(cfg.get('agent_name') or 'hermes'))}")
    clean_user_id = str(user_id or "").strip()
    if (
        clean_user_id
        and str(cfg.get("scope_by_user", True)).lower() not in ("0", "false", "no", "off")
    ):
        parts.append(f"user_{_normalize_name(clean_user_id)}")
    if not parts:
        parts.append("shared")
    return f"{base}__{_normalize_name('__'.join(parts))}"


def _print_result(command: list[str], result) -> int:
    print(f"Command: {' '.join(shlex.quote(p) for p in command)}")
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)
    return result.returncode


def cmd_status(args) -> None:
    cfg = _load_cfg()
    command, result = _run_cli(cfg, ["status"])
    code = _print_result(command, result)
    if code != 0:
        sys.exit(code)


def cmd_search(args) -> None:
    cfg = _load_cfg()
    cmd = ["search", args.query]
    if args.wing:
        cmd += ["--wing", args.wing]
    if args.room:
        cmd += ["--room", args.room]
    if args.results:
        cmd += ["--results", str(args.results)]
    command, result = _run_cli(cfg, cmd)
    code = _print_result(command, result)
    if code != 0:
        sys.exit(code)


def cmd_wakeup(args) -> None:
    cfg = _load_cfg()
    cmd = ["wake-up"]
    if args.wing:
        cmd += ["--wing", args.wing]
    else:
        cmd += ["--wing", _conversation_wing(cfg)]
    command, result = _run_cli(cfg, cmd)
    code = _print_result(command, result)
    if code != 0:
        sys.exit(code)


def cmd_flush(args) -> None:
    cfg = _load_cfg()
    transcript_dir = _transcript_dir(cfg)
    wing = _conversation_wing(cfg)
    command, result = _run_cli(
        cfg,
        ["mine", transcript_dir, "--mode", "convos", "--wing", wing, "--agent", str(cfg.get("agent_name") or "hermes")],
        timeout=float(cfg.get("session_sync_timeout_s") or 30.0),
    )
    code = _print_result(command, result)
    if code != 0:
        sys.exit(code)


def cmd_doctor(args) -> None:
    del args
    cfg = _load_cfg()
    command, _ = _base_command(cfg)
    transcript_dir = _transcript_dir(cfg)
    print("MemPalace doctor")
    print(f"Command: {' '.join(shlex.quote(p) for p in command)}")
    print(f"Transcript export dir: {transcript_dir}")
    print(f"Transcript export dir exists: {Path(transcript_dir).exists()}")
    checks = [
        ("status", ["status"], None),
        (
            "hook/session-start",
            ["hook", "run", "--hook", "session-start", "--harness", "codex"],
            {"session_id": "doctor", "transcript_path": str(Path(transcript_dir) / "doctor.jsonl"), "stop_hook_active": False},
        ),
    ]
    failed = False
    for label, args, payload in checks:
        command, result = _run_cli(cfg, args, stdin_json=payload)
        ok = result.returncode == 0
        print(f"[{'ok' if ok else 'fail'}] {label}")
        if result.stdout.strip():
            print(result.stdout.rstrip())
        if result.stderr.strip():
            print(result.stderr.rstrip(), file=sys.stderr)
        failed = failed or not ok
    if failed:
        sys.exit(1)


def register_cli(subparser) -> None:
    subs = subparser.add_subparsers(dest="mempalace_command")

    p_status = subs.add_parser("status", help="Show upstream MemPalace status output")
    p_status.set_defaults(func=cmd_status)

    p_doctor = subs.add_parser("doctor", help="Run basic MemPalace CLI health checks")
    p_doctor.set_defaults(func=cmd_doctor)

    p_search = subs.add_parser("search", help="Run MemPalace search")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--wing", default=None, help="Optional wing filter")
    p_search.add_argument("--room", default=None, help="Optional room filter")
    p_search.add_argument("--results", type=int, default=5, help="Result count")
    p_search.set_defaults(func=cmd_search)

    p_wakeup = subs.add_parser("wakeup", help="Show MemPalace wake-up context")
    p_wakeup.add_argument("--wing", default=None, help="Optional wing filter")
    p_wakeup.set_defaults(func=cmd_wakeup)

    p_flush = subs.add_parser("flush", help="Run a conversation mine over Hermes transcript exports")
    p_flush.set_defaults(func=cmd_flush)
