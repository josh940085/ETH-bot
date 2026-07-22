import os
from pathlib import Path

from runtime_config import REPO_DIR, load_local_env


DEFAULT_RUNTIME_DIR = REPO_DIR / ".runtime"
DEFAULT_BOT_DATA_DIR = DEFAULT_RUNTIME_DIR / "data"
DEFAULT_BOT_AI_DATA_DIR = DEFAULT_RUNTIME_DIR / "ai"


def _normalize_path(raw: str, base_dir: Path = REPO_DIR) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_bot_data_dir() -> Path:
    raw = str(os.getenv("BOT_DATA_DIR", "") or "").strip()
    if raw:
        return _normalize_path(raw)
    return DEFAULT_BOT_DATA_DIR


def _resolve_bot_ai_data_dir(bot_data_dir: Path) -> Path:
    raw = str(os.getenv("BOT_AI_DATA_DIR", "") or "").strip()
    if raw:
        return _normalize_path(raw)

    return DEFAULT_BOT_AI_DATA_DIR

load_local_env()

BOT_DATA_DIR = _resolve_bot_data_dir()
BOT_AI_DATA_DIR = _resolve_bot_ai_data_dir(BOT_DATA_DIR)


def data_path(*parts: str) -> Path:
    return BOT_DATA_DIR.joinpath(*parts)


def ai_data_path(*parts: str) -> Path:
    return BOT_AI_DATA_DIR.joinpath(*parts)


def ensure_parent_dir(path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
