#!/usr/bin/env python3
import atexit
import datetime
import json
import os
import signal
import subprocess
import sys
import time
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None
from runtime_config import load_local_env
from runtime_paths import REPO_DIR, ai_data_path, data_path
from telegram import (
    configure_token,
    consume_restart_request,
    poll_telegram_commands,
)

ETH_FILE = REPO_DIR / "eth.py"
BACKTEST_FILE = REPO_DIR / "backtest.py"
HISTORICAL_BACKTEST_FILE = REPO_DIR / "historical_backtest.py"
MONTHLY_KLINE_DOWNLOAD_FILE = REPO_DIR / "monthly_kline_download.py"
MAINTENANCE_FILE = REPO_DIR / "maintenance.py"
RESTART_DELAY_SEC = 2
SUPERVISOR_RESTART_EXIT_CODE = 75
SUPERVISOR_STOP_EXIT_CODE = 76
POLL_INTERVAL_SEC = 1
BACKTEST_SUMMARY_PATH = data_path("backtest_latest_summary.json")
BACKTEST_TRADES_PATH = data_path("backtest_latest_trades.csv")
BACKTEST_LEARN_PATH = ai_data_path("backtest_ai_data.csv")
MAINTENANCE_REPORT_PATH = data_path("maintenance_latest_report.json")
HISTORICAL_BACKTEST_REPORT_PATH = data_path("historical_backtest_latest_report.json")
MONTHLY_KLINE_DOWNLOAD_REPORT_PATH = data_path("monthly_kline_download_latest.json")
POSITION_PANEL_PATH = data_path("docs/position.json")
SUPERVISOR_LOCK_PATH = data_path(".program_supervisor.lock")
SUPERVISOR_LOCK_FH = None


def _real_execution_busy(env=None) -> bool:
    env = env or os.environ
    priority_enabled = str(env.get("REAL_ORDER_PRIORITY_ENABLED", "1") or "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
    real_copy_enabled = str(env.get("BINANCE_REAL_COPY_ENABLED", "0") or "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if not (priority_enabled and real_copy_enabled):
        return False
    try:
        payload = json.loads(POSITION_PANEL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("open") and str(payload.get("position_source") or "binance") == "binance":
        return True
    return str(payload.get("strategy_execution_status") or "") in {"pending_confirmation", "submitting"}


def _preempt_background_process_for_real_order(proc, label: str) -> bool:
    if proc is None or proc.poll() is not None:
        return False
    if getattr(proc, "_real_order_preempt_sent", False):
        return False
    proc._real_order_preempt_sent = True
    proc.terminate()
    print(f"⏸️ 實單第一順位：停止背景{label}，先保留資源給實單管理")
    return True


def _run_git_command(args, timeout=60):
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )


def _get_sync_target():
    upstream = _run_git_command(
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        timeout=10,
    )
    ref = (upstream.stdout or "").strip()
    if upstream.returncode == 0 and "/" in ref:
        remote, branch = ref.split("/", 1)
        return remote, branch

    branch_result = _run_git_command(["branch", "--show-current"], timeout=10)
    branch = (branch_result.stdout or "").strip()
    if branch:
        return "origin", branch

    return None


def sync_repo() -> bool:
    try:
        target = _get_sync_target()
        if not target:
            print("⚠️ 無法判斷目前分支的同步目標，略過 git pull")
            return False

        remote, branch = target
        dirty = _run_git_command(["status", "--porcelain"], timeout=10)
        if dirty.returncode == 0 and (dirty.stdout or "").strip():
            print("⚠️ 工作樹有未提交變更，git pull 可能失敗")

        result = _run_git_command(["pull", "--ff-only", remote, branch], timeout=60)
        print(f"📥 同步目標: {remote}/{branch}")
        print("📥 同步結果:")
        print((result.stdout or "").strip() or "(no output)")
        if result.returncode != 0:
            print("⚠️ 同步失敗，將使用本機目前的 eth.py 重新啟動")
        return result.returncode == 0
    except Exception as e:
        print(f"⚠️ 同步失敗: {e}")
        return False


def _release_supervisor_lock():
    global SUPERVISOR_LOCK_FH
    if SUPERVISOR_LOCK_FH is None:
        return

    try:
        if fcntl is not None:
            fcntl.flock(SUPERVISOR_LOCK_FH.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass

    try:
        SUPERVISOR_LOCK_FH.close()
    except Exception:
        pass

    SUPERVISOR_LOCK_FH = None


def _acquire_supervisor_lock():
    global SUPERVISOR_LOCK_FH
    if SUPERVISOR_LOCK_FH is not None:
        return True

    lock_path = SUPERVISOR_LOCK_PATH
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")

    if fcntl is None:
        SUPERVISOR_LOCK_FH = fh
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        return True

    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        owner = fh.read().strip()
        try:
            fh.close()
        except Exception:
            pass
        owner_text = f" PID={owner}" if owner else ""
        print(f"⚠️ 已有另一個 program.py supervisor 在執行，略過啟動{owner_text}")
        return False

    SUPERVISOR_LOCK_FH = fh
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    atexit.register(_release_supervisor_lock)
    return True


def _get_backtest_settings():
    enabled = str(os.getenv("BACKTEST_AUTO_ENABLED", "1") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    interval_sec = max(900.0, float(os.getenv("BACKTEST_INTERVAL_SEC", 6 * 3600)))
    startup_delay_sec = max(15.0, float(os.getenv("BACKTEST_STARTUP_DELAY_SEC", 90)))
    lookback_days = max(1, int(os.getenv("BACKTEST_LOOKBACK_DAYS", 14)))
    warmup_bars = max(200, int(os.getenv("BACKTEST_WARMUP_BARS", 1500)))
    market_source_preference = str(
        os.getenv("BACKTEST_MARKET_KLINE_SOURCE_PREFERENCE", "tradingview_first")
        or "tradingview_first"
    ).strip().lower()
    return {
        "enabled": enabled,
        "interval_sec": interval_sec,
        "startup_delay_sec": startup_delay_sec,
        "lookback_days": lookback_days,
        "warmup_bars": warmup_bars,
        "market_source_preference": market_source_preference,
    }


def _parse_daily_schedule(raw_value, default_hour=4, default_minute=30):
    raw = str(raw_value or "").strip()
    if ":" not in raw:
        return default_hour, default_minute

    try:
        hour_text, minute_text = raw.split(":", 1)
        hour = min(23, max(0, int(hour_text)))
        minute = min(59, max(0, int(minute_text)))
        return hour, minute
    except Exception:
        return default_hour, default_minute


def _read_latest_maintenance_report():
    if not MAINTENANCE_REPORT_PATH.exists():
        return {}

    try:
        payload = json.loads(MAINTENANCE_REPORT_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_latest_maintenance_finished_at():
    payload = _read_latest_maintenance_report()
    raw_value = str(payload.get("finished_at", "") or "").strip()
    if not raw_value:
        return None

    try:
        return datetime.datetime.fromisoformat(raw_value).astimezone()
    except Exception:
        return None


def _compute_next_daily_run_ts(hour, minute, now_ts=None):
    now_dt = datetime.datetime.fromtimestamp(now_ts or time.time()).astimezone()
    target_dt = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target_dt <= now_dt:
        target_dt += datetime.timedelta(days=1)
    return target_dt.timestamp()


def _get_maintenance_settings():
    enabled = str(os.getenv("MAINTENANCE_AUTO_ENABLED", "1") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    startup_delay_sec = max(30.0, float(os.getenv("MAINTENANCE_STARTUP_DELAY_SEC", 120)))
    daily_hour, daily_minute = _parse_daily_schedule(os.getenv("MAINTENANCE_TIME", "04:30"))
    smoke_backtest_days = max(1, int(os.getenv("MAINTENANCE_BACKTEST_DAYS", 3)))
    smoke_backtest_warmup_bars = max(200, int(os.getenv("MAINTENANCE_BACKTEST_WARMUP_BARS", 600)))
    notify = str(os.getenv("MAINTENANCE_NOTIFY", "1") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    package_auto_update = str(os.getenv("MAINTENANCE_PACKAGE_AUTO_UPDATE", "1") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    return {
        "enabled": enabled,
        "startup_delay_sec": startup_delay_sec,
        "daily_hour": daily_hour,
        "daily_minute": daily_minute,
        "smoke_backtest_days": smoke_backtest_days,
        "smoke_backtest_warmup_bars": smoke_backtest_warmup_bars,
        "notify": notify,
        "package_auto_update": package_auto_update,
    }


def _get_historical_backtest_settings():
    enabled = str(os.getenv("HISTORICAL_BACKTEST_AUTO_ENABLED", "1") or "1").strip().lower() not in {
        "0", "false", "no", "off",
    }
    daily_hour, daily_minute = _parse_daily_schedule(os.getenv("HISTORICAL_BACKTEST_TIME", "03:00"), 3, 0)
    startup_delay_sec = max(30.0, float(os.getenv("HISTORICAL_BACKTEST_STARTUP_DELAY_SEC", 180)))
    return {
        "enabled": enabled,
        "daily_hour": daily_hour,
        "daily_minute": daily_minute,
        "startup_delay_sec": startup_delay_sec,
    }


def _get_monthly_kline_download_settings():
    enabled = str(os.getenv("MONTHLY_KLINE_AUTO_ENABLED", "1") or "1").strip().lower() not in {
        "0", "false", "no", "off",
    }
    daily_hour, daily_minute = _parse_daily_schedule(os.getenv("MONTHLY_KLINE_TIME", "10:00"), 10, 0)
    return {
        "enabled": enabled,
        "daily_hour": daily_hour,
        "daily_minute": daily_minute,
        "startup_delay_sec": max(30.0, float(os.getenv("MONTHLY_KLINE_STARTUP_DELAY_SEC", 150))),
        "retry_sec": max(900.0, float(os.getenv("MONTHLY_KLINE_RETRY_SEC", 6 * 3600))),
    }


def _previous_utc_month_key(now_ts=None):
    now_utc = datetime.datetime.fromtimestamp(now_ts or time.time(), tz=datetime.timezone.utc)
    first = datetime.datetime(now_utc.year, now_utc.month, 1, tzinfo=datetime.timezone.utc)
    previous = first - datetime.timedelta(days=1)
    return f"{previous.year:04d}-{previous.month:02d}"


def _next_monthly_schedule_ts(hour, minute, now_ts=None):
    now_dt = datetime.datetime.fromtimestamp(now_ts or time.time()).astimezone()
    if now_dt.month == 12:
        target = now_dt.replace(year=now_dt.year + 1, month=1, day=1, hour=hour, minute=minute, second=0, microsecond=0)
    else:
        target = now_dt.replace(month=now_dt.month + 1, day=1, hour=hour, minute=minute, second=0, microsecond=0)
    return target.timestamp()


def _read_monthly_kline_report():
    try:
        payload = json.loads(MONTHLY_KLINE_DOWNLOAD_REPORT_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _compute_initial_monthly_kline_ts(settings, now_ts=None):
    now_ts = now_ts or time.time()
    now_dt = datetime.datetime.fromtimestamp(now_ts).astimezone()
    report = _read_monthly_kline_report()
    if report.get("success") and str(report.get("target_month") or "") == _previous_utc_month_key(now_ts):
        return _next_monthly_schedule_ts(settings["daily_hour"], settings["daily_minute"], now_ts)
    scheduled = now_dt.replace(day=1, hour=settings["daily_hour"], minute=settings["daily_minute"], second=0, microsecond=0)
    if now_dt >= scheduled:
        return now_ts + settings["startup_delay_sec"]
    return scheduled.timestamp()


def _compute_initial_historical_backtest_ts(settings):
    now_ts = time.time()
    now_dt = datetime.datetime.fromtimestamp(now_ts).astimezone()
    scheduled_dt = now_dt.replace(hour=settings["daily_hour"], minute=settings["daily_minute"], second=0, microsecond=0)
    try:
        payload = json.loads(HISTORICAL_BACKTEST_REPORT_PATH.read_text(encoding="utf-8"))
        finished_at = datetime.datetime.fromisoformat(str(payload.get("finished_at") or "")).astimezone()
        completed_today = bool(payload.get("success")) and finished_at.date() == now_dt.date()
    except Exception:
        completed_today = False
    if completed_today:
        return _compute_next_daily_run_ts(settings["daily_hour"], settings["daily_minute"], now_ts)
    if now_dt >= scheduled_dt:
        return now_ts + settings["startup_delay_sec"]
    return scheduled_dt.timestamp()


def _compute_initial_maintenance_ts(settings):
    now_ts = time.time()
    now_dt = datetime.datetime.fromtimestamp(now_ts).astimezone()
    scheduled_dt = now_dt.replace(
        hour=settings["daily_hour"],
        minute=settings["daily_minute"],
        second=0,
        microsecond=0,
    )
    last_finished_at = _read_latest_maintenance_finished_at()

    if last_finished_at is not None and last_finished_at.date() == now_dt.date():
        return _compute_next_daily_run_ts(settings["daily_hour"], settings["daily_minute"], now_ts)

    if now_dt >= scheduled_dt:
        return now_ts + settings["startup_delay_sec"]

    return scheduled_dt.timestamp()


def _start_backtest_process(env, settings):
    if not BACKTEST_FILE.exists():
        return None

    cmd = [
        sys.executable,
        str(BACKTEST_FILE),
        "--days",
        str(settings["lookback_days"]),
        "--warmup-bars",
        str(settings["warmup_bars"]),
        "--summary-out",
        str(BACKTEST_SUMMARY_PATH),
        "--trades-out",
        str(BACKTEST_TRADES_PATH),
        "--learn-out",
        str(BACKTEST_LEARN_PATH),
    ]
    print(f"🧪 啟動定時回測: {' '.join(cmd)}")
    backtest_env = env.copy()
    backtest_env["MARKET_KLINE_SOURCE_PREFERENCE"] = settings["market_source_preference"]
    return subprocess.Popen(
        cmd,
        cwd=str(REPO_DIR),
        env=backtest_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _start_maintenance_process(env, settings):
    if not MAINTENANCE_FILE.exists():
        return None

    cmd = [
        sys.executable,
        str(MAINTENANCE_FILE),
        "--report-out",
        str(MAINTENANCE_REPORT_PATH),
        "--smoke-backtest-days",
        str(settings["smoke_backtest_days"]),
        "--smoke-backtest-warmup-bars",
        str(settings["smoke_backtest_warmup_bars"]),
    ]
    if not settings["notify"]:
        cmd.append("--no-notify")
    if settings.get("package_auto_update", True):
        cmd.append("--update-packages")

    print(f"🛠️ 啟動每日巡檢: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=str(REPO_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _start_historical_backtest_process(env):
    if not HISTORICAL_BACKTEST_FILE.exists():
        return None
    cmd = [sys.executable, str(HISTORICAL_BACKTEST_FILE)]
    print(f"📚 啟動每日歷年回測: {' '.join(cmd)}")
    # Keep output attached to program.log so a multi-year run cannot block on a full PIPE buffer.
    return subprocess.Popen(cmd, cwd=str(REPO_DIR), env=env)


def _start_monthly_kline_download_process(env):
    if not MONTHLY_KLINE_DOWNLOAD_FILE.exists():
        return None
    cmd = [sys.executable, str(MONTHLY_KLINE_DOWNLOAD_FILE)]
    print(f"📥 啟動上月K線下載: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=str(REPO_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _terminate_process(proc, timeout=10):
    if proc is None or proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def run_once() -> int:
    load_local_env(overwrite=True, names=(".env",))
    configure_token(os.getenv("TELEGRAM_TOKEN", ""))
    backtest_settings = _get_backtest_settings()
    maintenance_settings = _get_maintenance_settings()
    historical_backtest_settings = _get_historical_backtest_settings()
    monthly_kline_settings = _get_monthly_kline_download_settings()

    env = os.environ.copy()
    env["BOT_SUPERVISOR"] = "1"

    cmd = [sys.executable, str(ETH_FILE)]
    print(f"🚀 啟動策略: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, cwd=str(REPO_DIR), env=env)
    backtest_proc = None
    maintenance_proc = None
    historical_backtest_proc = None
    monthly_kline_proc = None
    next_backtest_ts = time.time() + backtest_settings["startup_delay_sec"]
    next_maintenance_ts = _compute_initial_maintenance_ts(maintenance_settings)
    next_historical_backtest_ts = _compute_initial_historical_backtest_ts(historical_backtest_settings)
    next_monthly_kline_ts = _compute_initial_monthly_kline_ts(monthly_kline_settings)
    shutdown_requested = {"value": False, "signal": None}

    def _forward_signal(sig, _frame):
        shutdown_requested["value"] = True
        shutdown_requested["signal"] = sig
        if proc.poll() is None:
            proc.send_signal(sig)
        if backtest_proc is not None and backtest_proc.poll() is None:
            backtest_proc.send_signal(sig)
        if maintenance_proc is not None and maintenance_proc.poll() is None:
            maintenance_proc.send_signal(sig)
        if historical_backtest_proc is not None and historical_backtest_proc.poll() is None:
            historical_backtest_proc.send_signal(sig)
        if monthly_kline_proc is not None and monthly_kline_proc.poll() is None:
            monthly_kline_proc.send_signal(sig)

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    while True:
        code = proc.poll()
        if code is not None:
            _terminate_process(backtest_proc)
            _terminate_process(maintenance_proc)
            _terminate_process(historical_backtest_proc)
            _terminate_process(monthly_kline_proc)
            if shutdown_requested["value"]:
                return SUPERVISOR_STOP_EXIT_CODE
            return code

        real_execution_busy = _real_execution_busy(env)
        if real_execution_busy:
            _preempt_background_process_for_real_order(backtest_proc, "回測")
            _preempt_background_process_for_real_order(historical_backtest_proc, "歷年回測")
            _preempt_background_process_for_real_order(monthly_kline_proc, "K線下載")

        if backtest_proc is not None:
            backtest_code = backtest_proc.poll()
            if backtest_code is not None:
                output = ""
                try:
                    stdout, _ = backtest_proc.communicate(timeout=2)
                    output = (stdout or "").strip()
                except Exception:
                    output = ""
                print(f"🧪 定時回測結束，exit_code={backtest_code}")
                if output:
                    print(output)
                next_backtest_ts = time.time() + backtest_settings["interval_sec"]
                backtest_proc = None
        elif backtest_settings["enabled"] and not real_execution_busy and maintenance_proc is None and historical_backtest_proc is None and monthly_kline_proc is None and time.time() >= next_backtest_ts:
            backtest_proc = _start_backtest_process(env, backtest_settings)
            next_backtest_ts = time.time() + backtest_settings["interval_sec"]

        if maintenance_proc is not None:
            maintenance_code = maintenance_proc.poll()
            if maintenance_code is not None:
                output = ""
                try:
                    stdout, _ = maintenance_proc.communicate(timeout=2)
                    output = (stdout or "").strip()
                except Exception:
                    output = ""
                print(f"🛠️ 每日巡檢結束，exit_code={maintenance_code}")
                if output:
                    print(output)
                next_maintenance_ts = _compute_next_daily_run_ts(
                    maintenance_settings["daily_hour"],
                    maintenance_settings["daily_minute"],
                )
                maintenance_proc = None
        elif (
            maintenance_settings["enabled"]
            and not real_execution_busy
            and backtest_proc is None
            and historical_backtest_proc is None
            and monthly_kline_proc is None
            and time.time() >= next_maintenance_ts
        ):
            maintenance_proc = _start_maintenance_process(env, maintenance_settings)

        if historical_backtest_proc is not None:
            historical_code = historical_backtest_proc.poll()
            if historical_code is not None:
                print(f"📚 每日歷年回測結束，exit_code={historical_code}")
                next_historical_backtest_ts = _compute_next_daily_run_ts(
                    historical_backtest_settings["daily_hour"], historical_backtest_settings["daily_minute"]
                )
                historical_backtest_proc = None
        elif (
            historical_backtest_settings["enabled"]
            and not real_execution_busy
            and backtest_proc is None
            and maintenance_proc is None
            and monthly_kline_proc is None
            and time.time() >= next_historical_backtest_ts
        ):
            historical_backtest_proc = _start_historical_backtest_process(env)

        if monthly_kline_proc is not None:
            monthly_code = monthly_kline_proc.poll()
            if monthly_code is not None:
                output = ""
                try:
                    stdout, _ = monthly_kline_proc.communicate(timeout=2)
                    output = (stdout or "").strip()
                except Exception:
                    output = ""
                print(f"📥 上月K線下載結束，exit_code={monthly_code}")
                if output:
                    print(output)
                if monthly_code == 0:
                    next_monthly_kline_ts = _next_monthly_schedule_ts(
                        monthly_kline_settings["daily_hour"], monthly_kline_settings["daily_minute"]
                    )
                else:
                    next_monthly_kline_ts = time.time() + monthly_kline_settings["retry_sec"]
                monthly_kline_proc = None
        elif (
            monthly_kline_settings["enabled"]
            and not real_execution_busy
            and backtest_proc is None
            and maintenance_proc is None
            and historical_backtest_proc is None
            and time.time() >= next_monthly_kline_ts
        ):
            monthly_kline_proc = _start_monthly_kline_download_process(env)

        poll_telegram_commands()

        if consume_restart_request():
            print("♻️ 收到 Telegram /restart 請求，停止子程序並重啟")
            _terminate_process(proc)
            _terminate_process(backtest_proc)
            _terminate_process(maintenance_proc)
            _terminate_process(historical_backtest_proc)
            _terminate_process(monthly_kline_proc)
            return SUPERVISOR_RESTART_EXIT_CODE

        time.sleep(POLL_INTERVAL_SEC)


def main():
    if not ETH_FILE.exists():
        print("❌ 找不到 eth.py，請確認檔案存在")
        raise SystemExit(1)

    if not _acquire_supervisor_lock():
        raise SystemExit(0)

    load_local_env(overwrite=True, names=(".env",))
    configure_token(os.getenv("TELEGRAM_TOKEN", ""))

    while True:
        try:
            exit_code = run_once()
            if exit_code == SUPERVISOR_STOP_EXIT_CODE:
                print("🛑 收到停止訊號，結束啟動器")
                break
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
