from pathlib import Path
import os


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_RUNTIME_DIR = REPO_DIR / ".runtime"
DEFAULT_BOT_DATA_DIR = DEFAULT_RUNTIME_DIR / "data"
DEFAULT_BOT_AI_DATA_DIR = DEFAULT_RUNTIME_DIR / "ai"
LEGACY_AI_DATA_DIR = Path("/Volumes/SSD/trading")


def _load_local_env():
    for name in (".env", "token.env"):
        path = REPO_DIR / name
        if not path.exists():
            continue

        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            continue


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

    if DEFAULT_BOT_AI_DATA_DIR.exists() or bot_data_dir == DEFAULT_BOT_DATA_DIR:
        return DEFAULT_BOT_AI_DATA_DIR

    if str(os.getenv("BOT_DATA_DIR", "") or "").strip():
        return bot_data_dir

    if LEGACY_AI_DATA_DIR.exists():
        return LEGACY_AI_DATA_DIR

    return bot_data_dir

_load_local_env()

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
