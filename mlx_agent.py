import json
import os
import sys

from runtime_config import is_truthy, load_local_env
from runtime_paths import REPO_DIR


def _install_fast_models_endpoint(server_module, model):
    """Avoid mlx-lm's per-request Hugging Face cache scan on /v1/models."""

    def handle_models_request(handler):
        handler._set_completion_headers(200)
        handler.end_headers()
        response = {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": handler.created,
                }
            ],
        }
        handler.wfile.write(json.dumps(response).encode("utf-8"))
        handler.wfile.flush()

    server_module.APIHandler.handle_models_request = handle_models_request


def main():
    load_local_env(names=(".env",))
    if not is_truthy(os.getenv("MLX_AGENT_ENABLED", "1")):
        print("MLX agent disabled by MLX_AGENT_ENABLED")
        return 0

    model = (os.getenv("MLX_MODEL", "Qwen/Qwen3-4B-MLX-4bit") or "").strip()
    host = (os.getenv("MLX_AGENT_HOST", "127.0.0.1") or "127.0.0.1").strip()
    port = (os.getenv("MLX_AGENT_PORT", "8080") or "8080").strip()
    prompt_cache_size = (os.getenv("MLX_PROMPT_CACHE_SIZE", "2") or "2").strip()
    prompt_cache_bytes = (os.getenv("MLX_PROMPT_CACHE_BYTES", "536870912") or "536870912").strip()
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
        "--prompt-cache-size",
        prompt_cache_size,
        "--prompt-cache-bytes",
        prompt_cache_bytes,
    ]
    from mlx_lm import server

    _install_fast_models_endpoint(server, model)
    server.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
