"""Run the pinned local n8n installation with a minimal, isolated environment."""

import json
import os
import secrets
import stat
import sys
from pathlib import Path

from runtime_config import REPO_DIR, read_local_env_values


N8N_HOME = REPO_DIR / ".runtime" / "n8n-home"
N8N_BINARY = REPO_DIR / ".runtime" / "n8n-package" / "node_modules" / ".bin" / "n8n"
NODE_BIN_DIR = Path("/opt/homebrew/opt/node@22/bin")
ENCRYPTION_KEY_FILE = N8N_HOME / ".encryption_key"
WEBHOOK_SECRET_FILE = N8N_HOME / ".webhook_secret"
N8N_SETTINGS_FILE = N8N_HOME / ".n8n" / "config"
ALLOWED_NOTIFICATION_SECRETS = ("TELEGRAM_TOKEN", "DISCORD_WEBHOOK", "DISCORD_NEWS")


def _ensure_encryption_key() -> str:
    N8N_HOME.mkdir(parents=True, exist_ok=True)
    if N8N_SETTINGS_FILE.exists():
        try:
            settings = json.loads(N8N_SETTINGS_FILE.read_text(encoding="utf-8"))
            existing_key = str(settings.get("encryptionKey") or "").strip()
            if existing_key:
                ENCRYPTION_KEY_FILE.write_text(existing_key + "\n", encoding="utf-8")
                ENCRYPTION_KEY_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
                return existing_key
        except Exception:
            pass

    if ENCRYPTION_KEY_FILE.exists():
        key = ENCRYPTION_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key

    key = secrets.token_hex(32)
    ENCRYPTION_KEY_FILE.write_text(key + "\n", encoding="utf-8")
    ENCRYPTION_KEY_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return key


def _ensure_webhook_secret() -> str:
    N8N_HOME.mkdir(parents=True, exist_ok=True)
    if WEBHOOK_SECRET_FILE.exists():
        secret = WEBHOOK_SECRET_FILE.read_text(encoding="utf-8").strip()
        if secret:
            return secret

    secret = secrets.token_hex(32)
    WEBHOOK_SECRET_FILE.write_text(secret + "\n", encoding="utf-8")
    WEBHOOK_SECRET_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return secret


def build_n8n_environment():
    webhook_secret = _ensure_webhook_secret()
    env = {
        "HOME": os.environ.get("HOME", str(REPO_DIR)),
        "PATH": f"{NODE_BIN_DIR}:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "N8N_USER_FOLDER": str(N8N_HOME),
        "N8N_ENCRYPTION_KEY": _ensure_encryption_key(),
        "ETH_BOT_N8N_WEBHOOK_SECRET": webhook_secret,
        "N8N_HOST": "127.0.0.1",
        "N8N_LISTEN_ADDRESS": "127.0.0.1",
        "N8N_PORT": "5678",
        "N8N_PROTOCOL": "http",
        "N8N_EDITOR_BASE_URL": "http://127.0.0.1:5678/",
        "N8N_WEBHOOK_URL": "http://127.0.0.1:5678/",
        "GENERIC_TIMEZONE": "Asia/Taipei",
        "TZ": "Asia/Taipei",
        "N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS": "true",
        "N8N_DIAGNOSTICS_ENABLED": "false",
        "N8N_PERSONALIZATION_ENABLED": "false",
        "N8N_AI_ENABLED": "false",
        "N8N_BLOCK_ENV_ACCESS_IN_NODE": "false",
        "N8N_SECURE_COOKIE": "false",
        "N8N_PYTHON_ENABLED": "false",
        "N8N_RUNNERS_MODE": "external",
        "N8N_RUNNERS_AUTH_TOKEN": webhook_secret,
        "N8N_UNVERIFIED_PACKAGES_ENABLED": "false",
        "N8N_COMMUNITY_PACKAGES_ENABLED": "false",
        "N8N_PUBLIC_API_DISABLED": "true",
        "N8N_TEMPLATES_ENABLED": "false",
        "N8N_VERSION_NOTIFICATIONS_ENABLED": "false",
        "N8N_RUNNERS_TASK_TIMEOUT": "60",
        "N8N_COMPRESSION_NODE_MAX_DECOMPRESSED_SIZE_BYTES": "268435456",
        "N8N_COMPRESSION_NODE_MAX_ZIP_ENTRIES": "1000",
        "NODES_EXCLUDE": (
            '["n8n-nodes-base.openAi","@n8n/n8n-nodes-langchain.openAi",'
            '"@n8n/n8n-nodes-langchain.openAiAssistant",'
            '"@n8n/n8n-nodes-langchain.lmOpenAi",'
            '"@n8n/n8n-nodes-langchain.lmChatOpenAi",'
            '"@n8n/n8n-nodes-langchain.lmChatAzureOpenAi",'
            '"@n8n/n8n-nodes-langchain.embeddingsOpenAi",'
            '"@n8n/n8n-nodes-langchain.embeddingsAzureOpenAi"]'
        ),
        "EXECUTIONS_DATA_PRUNE": "true",
        "EXECUTIONS_DATA_MAX_AGE": "168",
        "EXECUTIONS_DATA_SAVE_ON_SUCCESS": "none",
        "EXECUTIONS_DATA_SAVE_ON_ERROR": "none",
        "EXECUTIONS_DATA_SAVE_MANUAL_EXECUTIONS": "false",
    }

    local_values = read_local_env_values(names=(".env",))
    for name in ALLOWED_NOTIFICATION_SECRETS:
        value = str(local_values.get(name) or os.getenv(name, "") or "").strip()
        if value:
            env[name] = value
    return env


def main():
    if not N8N_BINARY.exists():
        raise SystemExit(
            "n8n 尚未安裝；請執行 /opt/homebrew/opt/node@22/bin/npm "
            "install --prefix .runtime/n8n-package n8n@2.31.5"
        )

    args = sys.argv[1:] or ["start"]
    os.execve(str(N8N_BINARY), [str(N8N_BINARY), *args], build_n8n_environment())


if __name__ == "__main__":
    main()
