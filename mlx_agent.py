import os
import runpy
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent


def _load_local_env():
    env_path = REPO_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("\"'")


def main():
    _load_local_env()
    if str(os.getenv("MLX_AGENT_ENABLED", "1")).strip().lower() not in {"1", "true", "yes", "on"}:
        print("MLX agent disabled by MLX_AGENT_ENABLED")
        return 0

    model = (os.getenv("MLX_MODEL", "Qwen/Qwen3-4B-MLX-4bit") or "").strip()
    host = (os.getenv("MLX_AGENT_HOST", "127.0.0.1") or "127.0.0.1").strip()
    port = (os.getenv("MLX_AGENT_PORT", "8080") or "8080").strip()
    os.environ.setdefault("HF_HOME", str(REPO_DIR / ".runtime" / "ai" / "huggingface"))

    sys.argv = [
        "mlx_lm.server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        port,
        "--chat-template-args",
        '{"enable_thinking":false}',
    ]
    runpy.run_module("mlx_lm.server", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
