#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
ETH_FILE = REPO_DIR / "eth.py"
RESTART_DELAY_SEC = 2


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
        if key and key not in os.environ:
            os.environ[key] = value


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

    return proc.wait()


def main():
    if not ETH_FILE.exists():
        print("❌ 找不到 eth.py，請確認檔案存在")
        raise SystemExit(1)

    load_local_env()

    while True:
        try:
            exit_code = run_once()
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
