#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from runtime_paths import REPO_DIR, data_path
from telegram import configure_token, get_notification_chat_ids, load_local_env, send_message


DEFAULT_REPORT_PATH = data_path("maintenance_latest_report.json")
CONFLICT_START_RE = re.compile(r"^<<<<<<< .*$", re.MULTILINE)
CONFLICT_END_RE = re.compile(r"^>>>>>>> .*$", re.MULTILINE)
REPORT_PREFIX = "REPORT_JSON="
TEXT_SCAN_FILES = (
    "Dockerfile",
    "README.md",
    "requirements.txt",
    "requirements-realtime.txt",
)
JSON_AUTO_FIX_DEFAULTS = {
    data_path(".telegram_state.json"): {},
    data_path("news_stats_cache.json"): {},
    data_path("backtest_latest_summary.json"): {},
    data_path("docs", "position.json"): {
        "open": False,
        "ts": 0,
    },
}
TELEGRAM_BOT_COMMANDS = [
    {"command": "start", "description": "開始使用與顯示控制面板"},
    {"command": "help", "description": "顯示可用指令與說明"},
    {"command": "whoami", "description": "顯示你的 Telegram user id"},
    {"command": "settings", "description": "顯示跟單與控制面板設定"},
    {"command": "panel", "description": "開啟倉位面板與控制面板"},
    {"command": "menu", "description": "顯示控制面板"},
    {"command": "follow", "description": "開啟或關閉跟單"},
    {"command": "sync", "description": "同步幣安倉位"},
    {"command": "tp", "description": "設定止盈，例如 /tp 2300"},
    {"command": "sl", "description": "設定止損，例如 /sl 2350"},
    {"command": "tpsl", "description": "同時設定止盈止損"},
    {"command": "ai", "description": "AI 市場分析"},
    {"command": "news", "description": "取得最新新聞"},
    {"command": "restart", "description": "重新啟動 bot"},
]


def _truncate_text(value, limit=3500):
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _extract_repair_lines(output, limit=8):
    keywords = ("♻️", "重建", "重新訓練", "修正", "修復")
    lines = []
    seen = set()
    for raw_line in (output or "").splitlines():
        line = str(raw_line or "").strip()
        if not line or not any(keyword in line for keyword in keywords):
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _telegram_api_request(token, method, http_method="GET", payload=None, timeout=15):
    response = requests.request(
        http_method,
        f"https://api.telegram.org/bot{token}/{method}",
        params=payload if http_method.upper() == "GET" else None,
        json=payload if http_method.upper() != "GET" else None,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not data.get("ok"):
        description = ""
        if isinstance(data, dict):
            description = str(data.get("description", "") or "").strip()
        raise RuntimeError(description or f"Telegram API {method} failed")
    return data.get("result")


def _normalize_private_chat_list(values, limit=5):
    cleaned = []
    seen = set()
    for value in values:
        try:
            text = str(int(str(value).strip()))
        except Exception:
            continue
        if text.startswith("-") or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned[-max(1, int(limit)) :]


def _is_https_url(value):
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme == "https" and bool(parsed.netloc)


def _normalize_bot_commands(commands):
    normalized = []
    for item in commands if isinstance(commands, list) else []:
        if not isinstance(item, dict):
            continue
        command = str(item.get("command", "") or "").strip()
        description = str(item.get("description", "") or "").strip()
        if not command:
            continue
        normalized.append({"command": command, "description": description})
    return normalized


def _check_telegram_policy_and_repair():
    load_local_env()
    token = configure_token(os.getenv("TELEGRAM_TOKEN", ""))
    if not token:
        raise RuntimeError("missing TELEGRAM_TOKEN")

    repair_details = []
    repaired = []

    bot_info = _telegram_api_request(token, "getMe", "GET")
    if not isinstance(bot_info, dict) or not bot_info.get("is_bot"):
        raise RuntimeError("Telegram token is invalid or does not belong to a bot")

    webhook_info = _telegram_api_request(token, "getWebhookInfo", "GET")
    webhook_url = ""
    if isinstance(webhook_info, dict):
        webhook_url = str(webhook_info.get("url", "") or "").strip()
    if webhook_url:
        _telegram_api_request(
            token,
            "deleteWebhook",
            "POST",
            {"drop_pending_updates": False},
        )
        repaired.append("webhook")
        repair_details.append(
            {
                "target": "telegram_webhook",
                "action": "delete_webhook_for_long_polling",
                "content": webhook_url,
            }
        )

    state_path = data_path(".telegram_state.json")
    state_payload = _read_json(state_path)
    if isinstance(state_payload, dict):
        updated_state = dict(state_payload)
        original_state = json.dumps(state_payload, ensure_ascii=False, sort_keys=True)
        normalized_notify = _normalize_private_chat_list(updated_state.get("notify_chat_ids", []))
        if updated_state.get("notify_chat_ids") != normalized_notify:
            updated_state["notify_chat_ids"] = normalized_notify

        last_private = updated_state.get("last_private_chat_id")
        try:
            last_private_text = str(int(str(last_private).strip()))
            if last_private_text.startswith("-"):
                raise ValueError("group id is not allowed")
            updated_state["last_private_chat_id"] = int(last_private_text)
        except Exception:
            if updated_state.get("last_private_chat_id") not in ("", None):
                updated_state["last_private_chat_id"] = ""

        normalized_state = json.dumps(updated_state, ensure_ascii=False, sort_keys=True)
        if normalized_state != original_state:
            _write_json_atomic(state_path, updated_state)
            repaired.append("telegram_state")
            repair_details.append(
                {
                    "target": state_path.name,
                    "action": "normalize_private_chat_targets",
                    "content": json.dumps(updated_state, ensure_ascii=False, indent=2),
                }
            )

    mini_app_url = str(os.getenv("TELEGRAM_MINI_APP_URL", "") or "").strip()
    if mini_app_url and not _is_https_url(mini_app_url):
        raise RuntimeError(
            "TELEGRAM_MINI_APP_URL must be HTTPS because Telegram WebAppInfo.url requires an HTTPS URL"
        )

    try:
        env_chat_id = str(int(str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()))
        if env_chat_id.startswith("-"):
            raise RuntimeError("TELEGRAM_CHAT_ID must be a private chat id, not a group/channel id")
    except ValueError:
        if str(os.getenv("TELEGRAM_CHAT_ID", "") or "").strip():
            raise RuntimeError("TELEGRAM_CHAT_ID must be a numeric private chat id")

    current_commands = _normalize_bot_commands(_telegram_api_request(token, "getMyCommands", "GET"))
    desired_commands = _normalize_bot_commands(TELEGRAM_BOT_COMMANDS)
    if current_commands != desired_commands:
        _telegram_api_request(
            token,
            "setMyCommands",
            "POST",
            {"commands": TELEGRAM_BOT_COMMANDS},
        )
        repair_details.append(
            {
                "target": "telegram_commands",
                "action": "sync_bot_commands",
                "content": json.dumps(TELEGRAM_BOT_COMMANDS, ensure_ascii=False, indent=2),
            }
        )
        repaired.append("commands")

    detail = f"bot=@{bot_info.get('username', '')} long_polling_ready=True commands={len(TELEGRAM_BOT_COMMANDS)}"
    return {
        "status": "fixed" if repaired else "ok",
        "detail": detail,
        "repaired": repaired,
        "repair_details": repair_details,
        "bot_username": bot_info.get("username"),
    }


def _collect_recent_telegram_delivery_events(state_payload, window_sec=86400):
    if not isinstance(state_payload, dict):
        return []

    raw_events = state_payload.get("telegram_delivery_events")
    if not isinstance(raw_events, list):
        return []

    now_ts = int(dt.datetime.now(dt.timezone.utc).timestamp())
    cutoff_ts = now_ts - max(60, int(window_sec))
    events = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        try:
            ts = int(item.get("ts", 0) or 0)
        except Exception:
            continue
        if ts < cutoff_ts:
            continue
        event = dict(item)
        event["ts"] = ts
        event["chat_id"] = str(event.get("chat_id", "") or "").strip()
        event["category"] = str(event.get("category", "") or "").strip()
        event["context"] = str(event.get("context", "") or "").strip()
        event["ok"] = bool(event.get("ok"))
        events.append(event)

    events.sort(key=lambda item: item.get("ts", 0))
    return events


def _count_peak_delivery_rate(events):
    total_counts = {}
    chat_counts = {}
    peak_total_per_sec = 0
    peak_per_chat_per_sec = 0

    for item in events:
        ts = int(item.get("ts", 0) or 0)
        total_counts[ts] = int(total_counts.get(ts, 0)) + 1
        peak_total_per_sec = max(peak_total_per_sec, total_counts[ts])

        chat_id = str(item.get("chat_id", "") or "").strip()
        if chat_id:
            key = (chat_id, ts)
            chat_counts[key] = int(chat_counts.get(key, 0)) + 1
            peak_per_chat_per_sec = max(peak_per_chat_per_sec, chat_counts[key])

    return peak_total_per_sec, peak_per_chat_per_sec


def _check_telegram_watch_risk():
    state_path = data_path(".telegram_state.json")
    state_payload = _read_json(state_path) or {}
    events = _collect_recent_telegram_delivery_events(state_payload, window_sec=86400)

    counts = {}
    for item in events:
        category = str(item.get("category", "") or "").strip() or "unknown"
        counts[category] = int(counts.get(category, 0)) + 1

    peak_total_per_sec, peak_per_chat_per_sec = _count_peak_delivery_rate(events)
    repaired = []
    repair_details = []

    removable_chat_ids = {
        str(item.get("chat_id", "") or "").strip()
        for item in events
        if str(item.get("category", "") or "").strip() in {"blocked_by_user", "chat_not_found", "user_deactivated"}
        and str(item.get("chat_id", "") or "").strip()
    }

    if removable_chat_ids and isinstance(state_payload, dict):
        updated_state = dict(state_payload)
        notify_ids = _normalize_private_chat_list(updated_state.get("notify_chat_ids", []))
        filtered_notify = [item for item in notify_ids if item not in removable_chat_ids]
        changed = notify_ids != filtered_notify
        if changed:
            updated_state["notify_chat_ids"] = filtered_notify

        last_private = str(updated_state.get("last_private_chat_id", "") or "").strip()
        if last_private in removable_chat_ids:
            updated_state["last_private_chat_id"] = ""
            changed = True

        if changed:
            _write_json_atomic(state_path, updated_state)
            repaired.append("stale_chat_targets")
            repair_details.append(
                {
                    "target": state_path.name,
                    "action": "remove_blocked_or_invalid_chat_targets",
                    "content": json.dumps(sorted(removable_chat_ids), ensure_ascii=False),
                }
            )

    risk_level = "low"
    likely_causes = []
    recommendations = []
    status = "fixed" if repaired else "ok"

    rate_limited = int(counts.get("rate_limited", 0))
    unauthorized = int(counts.get("unauthorized", 0))
    forbidden = int(counts.get("forbidden", 0))
    blocked = int(counts.get("blocked_by_user", 0))
    invalid_targets = int(counts.get("chat_not_found", 0) + counts.get("user_deactivated", 0))
    total_errors = sum(count for key, count in counts.items() if key != "ok")

    if unauthorized > 0:
        status = "error"
        risk_level = "high"
        likely_causes.append("Bot token invalid, revoked, or permissions changed.")
        recommendations.append("重新檢查 TELEGRAM_TOKEN，並用 BotFather 確認 bot 狀態。")
    if rate_limited > 0:
        status = "error"
        risk_level = "high"
        likely_causes.append("Recent sends hit Telegram Bot API rate limits (429).")
        recommendations.append("降低短時間批量推播與重試頻率，依 retry_after 退避。")
    if peak_total_per_sec > 25 and risk_level != "high":
        risk_level = "medium"
        likely_causes.append("Peak broadcast rate approached Telegram bulk-send limits.")
        recommendations.append("把大量通知分散到更長時間窗，避免逼近 30 msg/sec。")
    if peak_per_chat_per_sec > 1 and risk_level == "low":
        risk_level = "medium"
        likely_causes.append("Some chats received bursts faster than Telegram's 1 msg/sec guidance.")
        recommendations.append("避免對同一 chat 連續瞬發多則訊息。")
    if blocked > 0:
        if risk_level == "low":
            risk_level = "medium"
        likely_causes.append("Some recipients blocked the bot or stopped accepting messages.")
        recommendations.append("減少非必要推播，避免讓使用者主動封鎖或回報 spam。")
    if invalid_targets > 0:
        likely_causes.append("Some saved chat targets are no longer valid.")
    if forbidden > 0 and risk_level != "high":
        status = "error"
        risk_level = "medium"
        likely_causes.append("Telegram rejected some sends with 403; check chat rights or user block state.")
        recommendations.append("檢查是否對無權限或已封鎖 bot 的 chat 發送。")

    if not events:
        likely_causes.append("No Telegram delivery events were recorded in the last 24 hours.")
        recommendations.append("這不代表官方沒有觀察；只是最近沒有可分析的送訊樣本。")

    detail = (
        "official_watchlist=not_exposed_by_bot_api "
        f"risk={risk_level} "
        f"last24h_events={len(events)} "
        f"errors={total_errors} "
        f"rate_limits={rate_limited} "
        f"blocked={blocked} "
        f"invalid_targets={invalid_targets} "
        f"peak_total_per_sec={peak_total_per_sec} "
        f"peak_per_chat_per_sec={peak_per_chat_per_sec}"
    )

    if not repair_details and recommendations and status == "fixed":
        repair_details = [f"recommendation: {item}" for item in recommendations]

    return {
        "status": status,
        "detail": detail,
        "repaired": repaired,
        "repair_details": repair_details,
        "official_watchlist_check": "not_exposed_by_bot_api",
        "risk_level": risk_level,
        "last24h_events": len(events),
        "error_count": total_errors,
        "rate_limit_count": rate_limited,
        "blocked_count": blocked,
        "invalid_target_count": invalid_targets,
        "peak_total_per_sec": peak_total_per_sec,
        "peak_per_chat_per_sec": peak_per_chat_per_sec,
        "likely_causes": likely_causes,
        "recommendations": recommendations,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Daily code maintenance for ETH-bot.")
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_PATH), help="JSON report output path")
    parser.add_argument(
        "--smoke-backtest-days",
        type=int,
        default=max(1, int(os.getenv("MAINTENANCE_BACKTEST_DAYS", "3") or "3")),
        help="Lookback days for maintenance smoke backtest",
    )
    parser.add_argument(
        "--smoke-backtest-warmup-bars",
        type=int,
        default=max(200, int(os.getenv("MAINTENANCE_BACKTEST_WARMUP_BARS", "600") or "600")),
        help="Warmup bars for maintenance smoke backtest",
    )
    parser.add_argument("--skip-smoke-backtest", action="store_true", help="Skip the maintenance smoke backtest")
    parser.add_argument("--no-notify", action="store_true", help="Do not send Telegram summary")
    return parser.parse_args()


def _iso_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_json_atomic(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, target)


def _read_json(path):
    target = Path(path)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_python_files():
    return sorted(REPO_DIR.glob("*.py"))


def _run_command(cmd, timeout=180, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        cmd,
        cwd=str(REPO_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )


def _extract_report_payload(output):
    for line in reversed((output or "").splitlines()):
        if line.startswith(REPORT_PREFIX):
            raw = line[len(REPORT_PREFIX):].strip()
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
            except Exception:
                return None
    return None


def _check_py_compile():
    files = _collect_python_files()
    compiled = []
    for path in files:
        py_compile.compile(str(path), doraise=True)
        compiled.append(path.name)
    return {
        "status": "ok",
        "detail": f"compiled {len(compiled)} python files",
        "files": compiled,
    }


def _check_conflict_markers():
    findings = []
    scan_paths = [REPO_DIR / name for name in TEXT_SCAN_FILES]
    scan_paths.extend(_collect_python_files())

    seen = set()
    for path in scan_paths:
        if path in seen or not path.exists() or path.is_dir():
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue

        if CONFLICT_START_RE.search(text) and CONFLICT_END_RE.search(text):
            findings.append(path.name)

    if findings:
        raise RuntimeError(f"merge conflict markers found in: {', '.join(findings)}")

    return {
        "status": "ok",
        "detail": f"scanned {len(seen)} text files",
    }


def _check_runtime_json_and_repair():
    repaired = []
    checked = []
    repair_details = []

    for path, default_payload in JSON_AUTO_FIX_DEFAULTS.items():
        target = Path(path)
        checked.append(target.name)
        if not target.exists():
            continue

        payload = _read_json(target)
        if isinstance(payload, dict):
            continue

        fixed_payload = dict(default_payload)
        if target.name == "position.json":
            fixed_payload["ts"] = int(dt.datetime.now().timestamp())

        _write_json_atomic(target, fixed_payload)
        repaired.append(target.name)
        repair_details.append(
            {
                "target": target.name,
                "action": "reset_corrupted_json",
                "content": json.dumps(fixed_payload, ensure_ascii=False, indent=2),
            }
        )

    detail = f"checked {len(checked)} runtime json files"
    if repaired:
        detail += f"; repaired {len(repaired)} corrupted files"

    return {
        "status": "fixed" if repaired else "ok",
        "detail": detail,
        "repaired": repaired,
        "repair_details": repair_details,
    }


def _check_import_smoke():
    snippet = """
import importlib.util
import json
import os
os.environ["ETH_BOT_DISABLE_LIVE"] = "1"
import runtime_paths
import telegram
import backtest
import program
import eth
modules = [
    "runtime_paths",
    "telegram",
    "backtest",
    "program",
    "eth",
]
skipped = []
if importlib.util.find_spec("fastapi") and importlib.util.find_spec("uvicorn"):
    import panel_realtime_server
    modules.append("panel_realtime_server")
else:
    skipped.append("panel_realtime_server")
print("REPORT_JSON=" + json.dumps({
    "modules": modules,
    "skipped": skipped,
}, ensure_ascii=False))
"""
    result = _run_command([sys.executable, "-c", snippet], timeout=180)
    payload = _extract_report_payload(result.stdout)
    if result.returncode != 0 or not payload:
        output = (result.stdout or "").strip()
        raise RuntimeError(output or "import smoke failed")

    return {
        "status": "ok",
        "detail": (
            f"imported {len(payload.get('modules', []))} modules"
            + (
                f"; skipped {', '.join(payload.get('skipped', []))}"
                if payload.get("skipped")
                else ""
            )
        ),
        "modules": payload.get("modules", []),
        "skipped": payload.get("skipped", []),
    }


def _check_models_and_repair():
    snippet = """
import json
import os
os.environ["ETH_BOT_DISABLE_LIVE"] = "1"
import eth
eth.load_model()
if eth.model is None:
    eth.retrain_model()
eth.load_news_model()
payload = {
    "batch_model_loaded": bool(eth.model is not None),
    "online_initialized": bool(getattr(eth, "online_initialized", False)),
    "online_sample_count": int(getattr(eth, "online_sample_count", 0)),
    "news_model_loaded": bool(getattr(eth, "news_model", None) is not None),
}
print("REPORT_JSON=" + json.dumps(payload, ensure_ascii=False))
raise SystemExit(0 if payload["batch_model_loaded"] and payload["news_model_loaded"] else 1)
"""
    result = _run_command([sys.executable, "-c", snippet], timeout=300)
    payload = _extract_report_payload(result.stdout)
    if result.returncode != 0 or not payload:
        output = (result.stdout or "").strip()
        raise RuntimeError(output or "model health check failed")

    output = result.stdout or ""
    repaired = any(keyword in output for keyword in ("♻️", "重建", "重新訓練"))
    repair_details = _extract_repair_lines(output)
    if repaired and not repair_details:
        repair_details = ["model health auto-repair completed"]
    return {
        "status": "fixed" if repaired else "ok",
        "detail": (
            f"batch={payload.get('batch_model_loaded')} "
            f"online_samples={payload.get('online_sample_count')} "
            f"news={payload.get('news_model_loaded')}"
        ),
        "repaired": ["models"] if repaired else [],
        "repair_details": repair_details,
        **payload,
    }


def _check_smoke_backtest(days, warmup_bars):
    with tempfile.TemporaryDirectory(prefix="ethbot_maintenance_") as tmp_dir:
        summary_path = Path(tmp_dir) / "summary.json"
        trades_path = Path(tmp_dir) / "trades.csv"
        cmd = [
            sys.executable,
            str(REPO_DIR / "backtest.py"),
            "--days",
            str(max(1, int(days))),
            "--warmup-bars",
            str(max(200, int(warmup_bars))),
            "--summary-out",
            str(summary_path),
            "--trades-out",
            str(trades_path),
        ]
        result = _run_command(cmd, timeout=900, extra_env={"ETH_BOT_DISABLE_LIVE": "1"})
        if result.returncode != 0:
            output = (result.stdout or "").strip()
            raise RuntimeError(output or "smoke backtest failed")

        summary = _read_json(summary_path) or {}
        if not isinstance(summary, dict):
            raise RuntimeError("smoke backtest did not produce a summary")

    detail = (
        f"days={days} trades={summary.get('trades', 0)} "
        f"win_rate={summary.get('win_rate', 0)}% "
        f"return={summary.get('total_return_pct', 0)}%"
    )
    return {
        "status": "ok",
        "detail": detail,
        "summary": summary,
    }


def _run_check(name, fn):
    started_at = _iso_now()
    result = {
        "name": name,
        "started_at": started_at,
        "status": "ok",
        "detail": "",
    }
    try:
        payload = fn()
        if isinstance(payload, dict):
            result.update(payload)
            result["status"] = str(payload.get("status", "ok") or "ok")
        else:
            result["detail"] = str(payload or "")
    except Exception as exc:
        result["status"] = "error"
        result["detail"] = str(exc)
    result["finished_at"] = _iso_now()
    return result


def _build_notification_text(report):
    header = {
        "ok": "Daily maintenance OK",
        "fixed": "Daily maintenance fixed issues",
        "error": "Daily maintenance found errors",
    }.get(report.get("status"), "Daily maintenance finished")

    lines = [
        header,
        f"time: {report.get('finished_at', '')}",
    ]
    if report.get("auto_fix_count", 0) > 0:
        lines.append(f"auto_fixes: {report['auto_fix_count']}")

    for check in report.get("checks", []):
        lines.append(f"- {check.get('name')}: {check.get('status')} | {check.get('detail')}")

    return "\n".join(lines)


def _build_fix_detail_texts(report):
    messages = []
    finished_at = str(report.get("finished_at", "") or "").strip()

    for check in report.get("checks", []):
        check_status = str(check.get("status", "") or "")
        if check_status not in {"fixed", "error"}:
            continue

        repair_details = check.get("repair_details")
        if not isinstance(repair_details, list) or not repair_details:
            repaired = check.get("repaired") if isinstance(check.get("repaired"), list) else []
            likely_causes = check.get("likely_causes") if isinstance(check.get("likely_causes"), list) else []
            recommendations = check.get("recommendations") if isinstance(check.get("recommendations"), list) else []

            if repaired:
                repair_details = [f"repaired: {', '.join(str(item) for item in repaired)}"]
            elif likely_causes or recommendations:
                repair_details = []
                for item in likely_causes:
                    text = str(item or "").strip()
                    if text:
                        repair_details.append(f"likely_cause: {text}")
                for item in recommendations:
                    text = str(item or "").strip()
                    if text:
                        repair_details.append(f"recommendation: {text}")
            else:
                repair_details = [str(check.get("detail", "") or "").strip()]

        title = "🛠️ 修正內容" if check_status == "fixed" else "⚠️ 關注項目"
        lines = [
            f"{title} | {check.get('name')}",
            f"time: {finished_at}",
        ]

        for item in repair_details:
            if isinstance(item, dict):
                target = str(item.get("target", "") or "").strip()
                action = str(item.get("action", "") or "").strip()
                content = str(item.get("content", "") or "").strip()

                if target and action:
                    lines.append(f"- {target}: {action}")
                elif target:
                    lines.append(f"- {target}")
                elif action:
                    lines.append(f"- {action}")

                if content:
                    lines.append(content)
                continue

            detail_line = str(item or "").strip()
            if detail_line:
                lines.append(f"- {detail_line}")

        messages.append(_truncate_text("\n".join(lines)))

    return messages


def _send_report_notification(report):
    load_local_env()
    token = configure_token(os.getenv("TELEGRAM_TOKEN", ""))
    if not token:
        return False

    targets = get_notification_chat_ids()
    if not targets:
        return False

    texts = [_build_notification_text(report), *_build_fix_detail_texts(report)]
    for chat_id in targets:
        for text in texts:
            if text:
                send_message(chat_id, text, timeout=8, token=token)
    return True


def main():
    args = _parse_args()
    report = {
        "started_at": _iso_now(),
        "status": "ok",
        "checks": [],
        "auto_fix_count": 0,
        "notify_sent": False,
    }

    checks = [
        ("conflict_markers", _check_conflict_markers),
        ("runtime_json", _check_runtime_json_and_repair),
        ("py_compile", _check_py_compile),
        ("import_smoke", _check_import_smoke),
        ("telegram_policy", _check_telegram_policy_and_repair),
        ("telegram_watch_risk", _check_telegram_watch_risk),
        ("model_health", _check_models_and_repair),
    ]
    if not args.skip_smoke_backtest:
        checks.append(
            (
                "smoke_backtest",
                lambda: _check_smoke_backtest(
                    days=args.smoke_backtest_days,
                    warmup_bars=args.smoke_backtest_warmup_bars,
                ),
            )
        )

    for name, fn in checks:
        result = _run_check(name, fn)
        report["checks"].append(result)

    auto_fix_count = 0
    has_error = False
    has_fixed = False
    for item in report["checks"]:
        if item.get("status") == "error":
            has_error = True
        if item.get("status") == "fixed":
            has_fixed = True
            repaired = item.get("repaired") if isinstance(item.get("repaired"), list) else []
            auto_fix_count += max(1, len(repaired) or 1)

    report["auto_fix_count"] = auto_fix_count
    report["status"] = "error" if has_error else ("fixed" if has_fixed else "ok")
    report["finished_at"] = _iso_now()

    _write_json_atomic(args.report_out, report)

    if not args.no_notify:
        report["notify_sent"] = _send_report_notification(report)
        _write_json_atomic(args.report_out, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if has_error else 0)


if __name__ == "__main__":
    main()
