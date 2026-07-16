#!/usr/bin/env python3
"""Run the panel's long-range backtests sequentially and publish only complete results."""

import datetime as dt
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from telegram import REPO_DIR, data_path


BACKTEST_FILE = REPO_DIR / "backtest.py"
REPORT_PATH = data_path("historical_backtest_latest_report.json")
BACKTEST_DIR = data_path("backtests")
PERIODS = (
    ("2022", "2022-01-01", "2023-01-01", "backtest_2022_market_profile_try3"),
    ("2023", "2023-01-01", "2024-01-01", "backtest_2023_market_profile_try3"),
    ("2024", "2024-01-01", "2025-01-01", "backtest_2024_market_profile_try3"),
    ("2025", "2025-01-01", "2026-01-01", "backtest_2025_market_profile_try5"),
    ("2026H1", "2026-01-01", "2026-07-01", "backtest_2026h1_market_profile_try5"),
)


def _write_report(payload):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = REPORT_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, REPORT_PATH)


def _load_summary(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"無法讀取回測摘要 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"回測摘要格式錯誤: {path}")
    return payload


def _summary_metrics(payload):
    try:
        trades = int(payload.get("trades"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("回測摘要缺少有效 trades") from exc
    try:
        final_equity = float(payload.get("final_equity"))
    except (TypeError, ValueError):
        try:
            final_equity = 1.0 + float(payload.get("total_return_pct")) / 100.0
        except (TypeError, ValueError) as exc:
            raise RuntimeError("回測摘要缺少有效 final_equity/total_return_pct") from exc
    if trades < 0 or not math.isfinite(final_equity) or final_equity <= 0:
        raise RuntimeError("回測摘要的 trades/final_equity 無效")
    return {"trades": trades, "final_equity": final_equity}


def _aggregate_metrics(period_metrics):
    total_trades = sum(int(item["trades"]) for item in period_metrics.values())
    compound_equity = math.prod(float(item["final_equity"]) for item in period_metrics.values())
    return {
        "trades": total_trades,
        "compound_return_pct": (compound_equity - 1.0) * 100.0,
    }


def _evaluate_candidate(candidate_summaries, baseline_summaries, min_return_improvement_pct=0.0):
    labels = [period[0] for period in PERIODS]
    if not baseline_summaries:
        candidate_metrics = {label: _summary_metrics(candidate_summaries[label]) for label in labels}
        return {
            "accepted": True,
            "bootstrap": True,
            "reason": "尚無正式基準，接受完整候選結果",
            "candidate": _aggregate_metrics(candidate_metrics),
            "baseline": None,
            "periods": {
                label: {"candidate": candidate_metrics[label], "baseline": None, "trade_count_preserved": True}
                for label in labels
            },
        }
    if set(baseline_summaries) != set(labels):
        return {
            "accepted": False,
            "bootstrap": False,
            "reason": "正式基準年份不完整，拒絕覆蓋",
            "missing_baseline_periods": sorted(set(labels) - set(baseline_summaries)),
        }

    candidate_metrics = {label: _summary_metrics(candidate_summaries[label]) for label in labels}
    baseline_metrics = {label: _summary_metrics(baseline_summaries[label]) for label in labels}
    candidate_total = _aggregate_metrics(candidate_metrics)
    baseline_total = _aggregate_metrics(baseline_metrics)
    min_gain = max(0.0, float(min_return_improvement_pct))
    return_improved = candidate_total["compound_return_pct"] > baseline_total["compound_return_pct"] + min_gain
    total_trades_preserved = candidate_total["trades"] >= baseline_total["trades"]
    periods = {}
    period_trades_preserved = True
    for label in labels:
        count_ok = candidate_metrics[label]["trades"] >= baseline_metrics[label]["trades"]
        period_trades_preserved = period_trades_preserved and count_ok
        periods[label] = {
            "candidate": candidate_metrics[label],
            "baseline": baseline_metrics[label],
            "trade_delta": candidate_metrics[label]["trades"] - baseline_metrics[label]["trades"],
            "trade_count_preserved": count_ok,
        }

    accepted = return_improved and total_trades_preserved and period_trades_preserved
    reasons = []
    if not return_improved:
        reasons.append("整體複利報酬未提高")
    if not total_trades_preserved:
        reasons.append("總單數下降")
    if not period_trades_preserved:
        reasons.append("至少一個年度單數下降")
    return {
        "accepted": accepted,
        "bootstrap": False,
        "reason": "符合報酬增加且單數不下降" if accepted else "、".join(reasons),
        "minimum_return_improvement_pct": min_gain,
        "return_improved": return_improved,
        "total_trade_count_preserved": total_trades_preserved,
        "all_period_trade_counts_preserved": period_trades_preserved,
        "candidate": candidate_total,
        "baseline": baseline_total,
        "return_delta_pct": candidate_total["compound_return_pct"] - baseline_total["compound_return_pct"],
        "trade_delta": candidate_total["trades"] - baseline_total["trades"],
        "periods": periods,
    }


def main():
    started_at = dt.datetime.now().astimezone().isoformat()
    results = []
    report = {"started_at": started_at, "finished_at": None, "success": False, "periods": results}
    _write_report(report)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="historical-backtest-", dir=str(BACKTEST_DIR)) as temp_dir:
        temp_dir = Path(temp_dir)
        candidate_files = {}
        candidate_summaries = {}
        for label, start, end, basename in PERIODS:
            summary_temp = temp_dir / f"{basename}_summary.json"
            trades_temp = temp_dir / f"{basename}_trades.csv"
            cmd = [
                sys.executable,
                str(BACKTEST_FILE),
                "--start", start,
                "--end", end,
                "--warmup-bars", os.getenv("HISTORICAL_BACKTEST_WARMUP_BARS", "1500"),
                "--data-source", os.getenv("HISTORICAL_BACKTEST_DATA_SOURCE", "binance-history"),
                "--summary-out", str(summary_temp),
                "--trades-out", str(trades_temp),
            ]
            print(f"📚 歷史回測 {label}: {' '.join(cmd)}", flush=True)
            result = subprocess.run(cmd, cwd=str(REPO_DIR), env=os.environ.copy(), check=False)
            period_result = {"period": label, "start": start, "end": end, "exit_code": result.returncode}
            results.append(period_result)
            if result.returncode != 0 or not summary_temp.exists() or not trades_temp.exists():
                report["failed_period"] = label
                report["finished_at"] = dt.datetime.now().astimezone().isoformat()
                _write_report(report)
                return result.returncode or 1

            try:
                summary_payload = _load_summary(summary_temp)
                period_result["candidate"] = _summary_metrics(summary_payload)
            except Exception as exc:
                period_result["validation_error"] = str(exc)
                report["failed_period"] = label
                report["finished_at"] = dt.datetime.now().astimezone().isoformat()
                _write_report(report)
                return 1
            candidate_summaries[label] = summary_payload
            candidate_files[label] = (summary_temp, trades_temp, basename)
            period_result["completed"] = True
            _write_report(report)

        baseline_summaries = {}
        for label, _start, _end, basename in PERIODS:
            baseline_path = BACKTEST_DIR / f"{basename}_summary.json"
            if baseline_path.exists():
                baseline_summaries[label] = _load_summary(baseline_path)

        try:
            acceptance = _evaluate_candidate(
                candidate_summaries,
                baseline_summaries,
                min_return_improvement_pct=os.getenv("HISTORICAL_BACKTEST_MIN_RETURN_IMPROVEMENT_PCT", "0"),
            )
        except Exception as exc:
            report["acceptance"] = {"accepted": False, "reason": f"接受條件檢查失敗: {exc}"}
            report["finished_at"] = dt.datetime.now().astimezone().isoformat()
            _write_report(report)
            return 1

        report["acceptance"] = acceptance
        report["published"] = False
        if not acceptance.get("accepted"):
            report["success"] = True
            report["finished_at"] = dt.datetime.now().astimezone().isoformat()
            _write_report(report)
            print(
                "🛡️ 歷年回測未達發布門檻，保留既有正式結果 | "
                f"原因={acceptance.get('reason')} | "
                f"報酬差={acceptance.get('return_delta_pct', 0):+.6f}% | "
                f"單數差={acceptance.get('trade_delta', 0):+d}",
                flush=True,
            )
            return 0

        for label, (summary_temp, trades_temp, basename) in candidate_files.items():
            os.replace(summary_temp, BACKTEST_DIR / f"{basename}_summary.json")
            os.replace(trades_temp, BACKTEST_DIR / f"{basename}_trades.csv")
            for period_result in results:
                if period_result["period"] == label:
                    period_result["published"] = True
                    break
        report["published"] = True

    report["success"] = True
    report["finished_at"] = dt.datetime.now().astimezone().isoformat()
    _write_report(report)
    print("✅ 歷年回測全部完成並更新 App 資料", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
