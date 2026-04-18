from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "mempalace.json"
DEFAULT_RECALL_LIMIT = 5
DEFAULT_SEARCH_TIMEOUT_S = 8.0
DEFAULT_PREFETCH_TIMEOUT_S = 8.0
DEFAULT_SESSION_SYNC_TIMEOUT_S = 30.0


def default_cli_command() -> str:
    return "mempalace"


def config_path(hermes_home: str) -> Path:
    return Path(hermes_home) / CONFIG_FILENAME


def load_provider_config(hermes_home: str) -> dict:
    path = config_path(hermes_home)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed to read MemPalace config from %s", path, exc_info=True)
        return {}


def save_provider_config(values: dict, hermes_home: str) -> None:
    path = config_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="mempalace-", suffix=".json.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(values, indent=2, ensure_ascii=False) + "\n")
        os.replace(temp_path, path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except OSError:
            pass


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)
