"""Shared environment-file loading and typed runtime settings."""

import os
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_FILES = (".env", "token.env")


def read_local_env_values(names=DEFAULT_ENV_FILES, repo_dir=REPO_DIR):
    values = {}
    base = Path(repo_dir)
    for name in names:
        path = base / str(name)
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key:
                    values[key] = value.strip().strip('"').strip("'")
        except Exception:
            continue
    return values


def load_local_env(*, overwrite=False, names=DEFAULT_ENV_FILES, repo_dir=REPO_DIR):
    values = read_local_env_values(names=names, repo_dir=repo_dir)
    for key, value in values.items():
        if overwrite or key not in os.environ:
            os.environ[key] = value
    return values


def env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def env_int(name, default):
    try:
        return int(str(os.getenv(name, default)).strip())
    except Exception:
        return int(default)


def env_bool(name, default=False):
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def is_truthy(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
