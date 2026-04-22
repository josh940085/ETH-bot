#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time
import json
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None
import requests
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
ETH_FILE = REPO_DIR / "eth.py"
TELEGRAM_STATE_FILE = REPO_DIR / ".telegram_state.json"
RESTART_DELAY_SEC = 2
SUPERVISOR_RESTART_EXIT_CODE = 75
TELEGRAM_TOKEN = ""
POLL_INTERVAL_SEC = 1
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "ETH-bot-supervisor/1.0"})


def load_local_env():
    """Load .env without extra dependency."""
    env_path = REPO_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            # 以 .env 為準，避免既有空值/舊值影響啟動器
            os.environ[key] = value


def _load_telegram_state() -> dict:
    return _read_telegram_state_locked()


def _parse_telegram_state(raw: str) -> dict:
    try:
        payload = json.loads(raw) if str(raw).strip() else {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_telegram_state_locked() -> dict:
    if not TELEGRAM_STATE_FILE.exists():
        return {}

    try:
        with TELEGRAM_STATE_FILE.open("r", encoding="utf-8") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            raw = fh.read()
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return _parse_telegram_state(raw)
    except Exception:
        return {}


def _update_telegram_state(mutator):
    try:
        TELEGRAM_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TELEGRAM_STATE_FILE.open("a+", encoding="utf-8") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

            fh.seek(0)
            payload = _parse_telegram_state(fh.read())
            result = mutator(payload)

            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.flush()
            os.fsync(fh.fileno())

            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return result
    except Exception as e:
        print(f"⚠️ 寫入 telegram state 失敗: {e}")
        return None


def _append_pending_command(chat_id, text, update_id):
    def _mutate(payload):
        queue = payload.get("pending_commands")
        if not isinstance(queue, list):
            queue = []

        queue.append(
            {
                "chat_id": chat_id,
                "text": text,
                "update_id": int(update_id),
                "ts": int(time.time()),
            }
        )
        payload["pending_commands"] = queue[-50:]
        payload["last_update_id"] = int(update_id)

    _update_telegram_state(_mutate)


def _append_pending_callback(chat_id, callback_data, callback_id, message_id, update_id):
    def _mutate(payload):
        queue = payload.get("pending_commands")
        if not isinstance(queue, list):
            queue = []

        queue.append(
            {
                "chat_id": chat_id,
                "text": f"__callback__:{callback_data}:{callback_id}:{message_id}",
                "update_id": int(update_id),
                "ts": int(time.time()),
            }
        )
        payload["pending_commands"] = queue[-50:]
        payload["last_update_id"] = int(update_id)

    _update_telegram_state(_mutate)


def _set_restart_requested(update_id):
    def _mutate(payload):
        payload["restart_requested"] = True
        payload["restart_requested_at"] = int(time.time())
        payload["last_update_id"] = int(update_id)

    _update_telegram_state(_mutate)


def _telegram_send_message(chat_id, text):
    if not TELEGRAM_TOKEN or chat_id is None:
        return

    try:
        HTTP_SESSION.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=5,
        )
    except Exception:
        pass


def poll_telegram_commands():
    if not TELEGRAM_TOKEN:
        return

    payload = _load_telegram_state()
    last_update_id = payload.get("last_update_id")

    params = {"timeout": 1}
    if last_update_id is not None:
        try:
            params["offset"] = int(last_update_id) + 1
        except Exception:
            pass

    try:
        res = HTTP_SESSION.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params=params,
            timeout=5,
        )
        res.raise_for_status()
        payload = res.json()
        updates = payload.get("result", []) if isinstance(payload, dict) else []
    except Exception as e:
        print(f"⚠️ 讀取 Telegram 更新失敗: {e}")
        updates = []

    for u in updates:
        update_id = u.get("update_id")
        msg = u.get("message", {})
        text = msg.get("text", "")
        chat_id = msg.get("chat", {}).get("id")

        if update_id is None:
            continue

        # 支援 inline button callback_query（轉發給 eth.py）
        cq = u.get("callback_query")
        if cq:
            cq_data = cq.get("data", "")
            cq_id = cq.get("id", "")
            cq_msg = cq.get("message", {})
            cq_msg_id = cq_msg.get("message_id")
            cq_chat_id = cq_msg.get("chat", {}).get("id")

            _append_pending_callback(
                cq_chat_id,
                cq_data,
                cq_id,
                cq_msg_id,
                update_id,
            )
            continue

        if not text:
            _update_telegram_state(lambda payload: payload.__setitem__("last_update_id", int(update_id)))
            continue

        if text.startswith("/restart"):
            _set_restart_requested(update_id)
            _telegram_send_message(chat_id, "♻️ 已收到 /restart，將由啟動器同步並重啟。")
            continue

        _append_pending_command(chat_id, text, update_id)


def consume_restart_request() -> bool:
    def _mutate(payload):
        if not payload.get("restart_requested"):
            return False
        payload["restart_requested"] = False
        payload["restart_requested_at"] = int(time.time())
        return True

    return bool(_update_telegram_state(_mutate))


def sync_repo() -> bool:
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only", "origin", "main"],
            cwd=str(REPO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
        print("📥 同步結果:")
        print((result.stdout or "").strip() or "(no output)")
        return result.returncode == 0
    except Exception as e:
        print(f"⚠️ 同步失敗: {e}")
        return False


def run_once() -> int:
    env = os.environ.copy()
    env["BOT_SUPERVISOR"] = "1"

    cmd = [sys.executable, str(ETH_FILE)]
    print(f"🚀 啟動策略: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, cwd=str(REPO_DIR), env=env)

    def _forward_signal(sig, _frame):
        if proc.poll() is None:
            proc.send_signal(sig)

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    while True:
        code = proc.poll()
        if code is not None:
            return code

        poll_telegram_commands()

        if consume_restart_request():
            print("♻️ 收到 Telegram /restart 請求，停止子程序並重啟")
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            return SUPERVISOR_RESTART_EXIT_CODE

        time.sleep(POLL_INTERVAL_SEC)


def main():
    global TELEGRAM_TOKEN

    if not ETH_FILE.exists():
        print("❌ 找不到 eth.py，請確認檔案存在")
        raise SystemExit(1)

    load_local_env()
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")

    while True:
        try:
            exit_code = run_once()
            if exit_code == SUPERVISOR_RESTART_EXIT_CODE:
                sync_repo()
            print(f"ℹ️ eth.py 已結束，exit_code={exit_code}，{RESTART_DELAY_SEC} 秒後重啟")
            time.sleep(RESTART_DELAY_SEC)
        except KeyboardInterrupt:
            print("\n🛑 收到中斷，停止啟動器")
            break
        except Exception as e:
            print(f"⚠️ 啟動器錯誤: {e}")
            time.sleep(RESTART_DELAY_SEC)


if __name__ == "__main__":
    main()
