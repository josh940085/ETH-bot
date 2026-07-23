"""Apply package updates to services after the live Binance position is safe."""

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from runtime_paths import REPO_DIR, data_path


POSITION_PATH = data_path("docs", "position.json")
REPORT_PATH = data_path("package_restart_latest.json")
SUPERVISORCTL = REPO_DIR / ".venv" / "bin" / "supervisorctl"
SERVICE_ORDER = ("n8n", "panel-realtime", "panel-tunnel", "mlx-agent", "eth-bot")


def _iso_now():
    return dt.datetime.now(ZoneInfo("Asia/Taipei")).isoformat()


def _position_busy():
    try:
        payload = json.loads(POSITION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return True
    if not isinstance(payload, dict):
        return True
    if payload.get("open") and str(payload.get("position_source") or "binance") == "binance":
        return True
    return str(payload.get("strategy_execution_status") or "") in {
        "pending_confirmation",
        "submitting",
    }


def _write_report(payload):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = REPORT_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, REPORT_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("services", nargs="*")
    parser.add_argument("--max-wait-sec", type=int, default=21600)
    args = parser.parse_args()

    requested = [name for name in SERVICE_ORDER if name in set(args.services)]
    report = {
        "started_at": _iso_now(),
        "status": "waiting",
        "requested": requested,
        "restarted": [],
    }
    _write_report(report)

    deadline = time.time() + max(0, int(args.max_wait_sec))
    while _position_busy() and time.time() < deadline:
        time.sleep(30)

    if _position_busy():
        report.update(
            status="pending",
            detail="實單持倉仍在，保留至下次安全巡檢重啟",
            finished_at=_iso_now(),
        )
        _write_report(report)
        return 0

    for service in requested:
        result = subprocess.run(
            [str(SUPERVISORCTL), "-c", str(REPO_DIR / "supervisord.conf"), "restart", service],
            cwd=str(REPO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
            check=False,
        )
        if result.returncode != 0:
            report.update(
                status="error",
                detail=f"{service} 重啟失敗: {str(result.stdout or '').strip()}",
                finished_at=_iso_now(),
            )
            _write_report(report)
            return 1
        report["restarted"].append(service)
        _write_report(report)
        time.sleep(3)

    report.update(
        status="ok",
        detail=f"安全重啟完成：{', '.join(report['restarted']) or '無需重啟'}",
        finished_at=_iso_now(),
    )
    _write_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
