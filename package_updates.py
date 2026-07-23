"""Inspect and safely update the packages used by the live ETH-bot stack."""

import datetime as dt
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

from runtime_paths import REPO_DIR, data_path


NPM_BIN = Path("/opt/homebrew/opt/node@22/bin/npm")
NODE_BIN_DIR = Path("/opt/homebrew/opt/node@22/bin")
BREW_BIN = Path("/opt/homebrew/bin/brew")
N8N_PACKAGE_ROOT = REPO_DIR / ".runtime" / "n8n-package"
N8N_PACKAGE_JSON = N8N_PACKAGE_ROOT / "node_modules" / "n8n" / "package.json"
PACKAGE_UPDATE_REPORT_PATH = data_path("package_updates_latest.json")
PACKAGE_SNAPSHOT_DIR = data_path("package_snapshots")


def _iso_now():
    return dt.datetime.now(ZoneInfo("Asia/Taipei")).isoformat()


def _run(cmd, *, timeout=900, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(item) for item in cmd],
        cwd=str(REPO_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )


def _write_json_atomic(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, target)


def _load_json_output(output, default):
    text = str(output or "").strip()
    try:
        return json.loads(text)
    except Exception:
        start = min((index for index in (text.find("["), text.find("{")) if index >= 0), default=-1)
        if start >= 0:
            try:
                return json.loads(text[start:])
            except Exception:
                pass
    return default


def _current_n8n_version():
    try:
        payload = json.loads(N8N_PACKAGE_JSON.read_text(encoding="utf-8"))
        return str(payload.get("version") or "").strip()
    except Exception:
        return ""


def _major_version(value):
    try:
        return int(str(value or "").strip().split(".", 1)[0])
    except Exception:
        return -1


def inspect_package_versions(*, refresh_brew=False):
    if refresh_brew and BREW_BIN.exists():
        _run([BREW_BIN, "update"], timeout=600)

    pip_result = _run(
        [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
        timeout=300,
    )
    pip_outdated = _load_json_output(pip_result.stdout, []) if pip_result.returncode == 0 else []

    current_n8n = _current_n8n_version()
    latest_n8n = ""
    if NPM_BIN.exists():
        npm_result = _run(
            [NPM_BIN, "view", "n8n", "version"],
            timeout=120,
            extra_env={"PATH": f"{NODE_BIN_DIR}:/usr/bin:/bin"},
        )
        if npm_result.returncode == 0:
            latest_n8n = str(npm_result.stdout or "").strip().splitlines()[0]

    brew_outdated = []
    if BREW_BIN.exists():
        brew_result = _run([BREW_BIN, "outdated", "--formula", "--json=v2"], timeout=300)
        brew_payload = _load_json_output(brew_result.stdout, {}) if brew_result.returncode == 0 else {}
        brew_outdated = list(brew_payload.get("formulae") or []) if isinstance(brew_payload, dict) else []

    return {
        "checked_at": _iso_now(),
        "python": {
            "version": sys.version.split()[0],
            "outdated": pip_outdated if isinstance(pip_outdated, list) else [],
        },
        "n8n": {
            "current": current_n8n,
            "latest": latest_n8n,
            "update_available": bool(current_n8n and latest_n8n and current_n8n != latest_n8n),
        },
        "homebrew": {
            "outdated": brew_outdated,
        },
    }


def _package_names(rows):
    names = []
    for item in rows if isinstance(rows, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _installed_python_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = ""
    return versions


def _validate_updated_runtime():
    tests = _run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
        timeout=600,
        extra_env={"ETH_BOT_DISABLE_LIVE": "1"},
    )
    if tests.returncode != 0:
        raise RuntimeError("更新後測試失敗: " + str(tests.stdout or "").strip()[-1200:])

    compile_result = _run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "eth.py",
            "news.py",
            "maintenance.py",
            "n8n_client.py",
            "n8n_service.py",
        ],
        timeout=180,
        extra_env={"PYTHONPYCACHEPREFIX": str(REPO_DIR / ".runtime" / "pycache")},
    )
    if compile_result.returncode != 0:
        raise RuntimeError("更新後語法檢查失敗: " + str(compile_result.stdout or "").strip()[-1200:])


def _snapshot_pip_environment():
    PACKAGE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = PACKAGE_SNAPSHOT_DIR / f"pip-freeze-{stamp}.txt"
    result = _run([sys.executable, "-m", "pip", "freeze"], timeout=180)
    if result.returncode != 0:
        raise RuntimeError("無法建立 pip 更新前快照")
    path.write_text(str(result.stdout or ""), encoding="utf-8")
    return path


def _restore_python_packages(snapshot_path):
    return _run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "-r", snapshot_path],
        timeout=1800,
    )


def check_and_update_packages(*, apply_updates=False):
    before = inspect_package_versions(refresh_brew=apply_updates)
    pip_names = _package_names(before["python"]["outdated"])
    brew_names = _package_names(before["homebrew"]["outdated"])
    n8n_current = str(before["n8n"].get("current") or "")
    n8n_latest = str(before["n8n"].get("latest") or "")
    update_count = len(pip_names) + len(brew_names) + int(bool(before["n8n"].get("update_available")))

    report = {
        "started_at": _iso_now(),
        "status": "ok",
        "detail": f"已檢查：Python {len(pip_names)}、Homebrew {len(brew_names)}、n8n {n8n_current or 'unknown'}",
        "before": before,
        "updated": [],
        "held": [],
        "repaired": [],
        "repair_details": [],
        "restart_services": [],
        "apply_updates": bool(apply_updates),
    }
    if not apply_updates or update_count == 0:
        report["finished_at"] = _iso_now()
        _write_json_atomic(PACKAGE_UPDATE_REPORT_PATH, report)
        return report

    pip_snapshot = None
    n8n_updated = False
    try:
        if pip_names:
            pip_versions_before = _installed_python_versions(pip_names)
            pip_snapshot = _snapshot_pip_environment()
            pip_update = _run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--upgrade-strategy",
                    "eager",
                    "-r",
                    "requirements.txt",
                ],
                timeout=1800,
            )
            if pip_update.returncode != 0:
                raise RuntimeError("Python 套件更新失敗: " + str(pip_update.stdout or "").strip()[-1200:])
            pip_versions_after = _installed_python_versions(pip_names)
            changed_python = [
                name
                for name in pip_names
                if pip_versions_before.get(name) != pip_versions_after.get(name)
            ]
            if changed_python:
                report["updated"].append({"manager": "pip", "packages": changed_python})
                report["restart_services"].extend(["panel-realtime", "mlx-agent", "eth-bot"])
            else:
                pip_snapshot = None

        if n8n_current and n8n_latest and n8n_current != n8n_latest:
            if _major_version(n8n_current) != _major_version(n8n_latest):
                report["held"].append(
                    f"n8n {n8n_current} -> {n8n_latest} 為主版本升級，保留人工遷移"
                )
            else:
                npm_update = _run(
                    [NPM_BIN, "install", "--prefix", N8N_PACKAGE_ROOT, f"n8n@{n8n_latest}"],
                    timeout=1800,
                    extra_env={"PATH": f"{NODE_BIN_DIR}:/usr/bin:/bin"},
                )
                if npm_update.returncode != 0:
                    raise RuntimeError("n8n 更新失敗: " + str(npm_update.stdout or "").strip()[-1200:])
                n8n_updated = True
                report["updated"].append(
                    {"manager": "npm", "package": "n8n", "from": n8n_current, "to": n8n_latest}
                )
                report["restart_services"].append("n8n")

        _validate_updated_runtime()

        if brew_names:
            brew_update = _run(
                [BREW_BIN, "upgrade", *brew_names],
                timeout=1800,
                extra_env={"HOMEBREW_NO_INSTALL_CLEANUP": "1"},
            )
            if brew_update.returncode != 0:
                raise RuntimeError("Homebrew 套件更新失敗: " + str(brew_update.stdout or "").strip()[-1200:])
            report["updated"].append({"manager": "homebrew", "packages": brew_names})
            if "cloudflared" in brew_names:
                report["restart_services"].append("panel-tunnel")
            if "node@22" in brew_names:
                report["restart_services"].append("n8n")
            if any(name in brew_names for name in ("python@3.11", "openssl@3", "libomp", "sqlite")):
                report["restart_services"].extend(
                    ["panel-realtime", "mlx-agent", "panel-tunnel", "eth-bot"]
                )
            _validate_updated_runtime()

        after = inspect_package_versions(refresh_brew=False)
        report["after"] = after
        remaining_python = _package_names(after["python"]["outdated"])
        if remaining_python:
            report["held"].append(
                "仍受相依限制保留：" + ", ".join(remaining_python[:20])
            )
        report["restart_services"] = list(dict.fromkeys(report["restart_services"]))
        report["repaired"] = [str(item.get("manager")) for item in report["updated"]]
        report["repair_details"] = [
            {
                "target": "package_versions",
                "action": "update_packages",
                "content": (
                    f"已更新 {len(report['updated'])} 組套件來源；"
                    f"待安全重啟：{', '.join(report['restart_services']) or '無'}"
                ),
            }
        ]
        report["status"] = "fixed" if report["updated"] else "ok"
        report["detail"] = (
            f"版本巡檢完成；更新來源={len(report['updated'])}；"
            f"保留項目={len(report['held'])}"
        )
    except Exception as exc:
        rollback = []
        if pip_snapshot is not None:
            restored = _restore_python_packages(pip_snapshot)
            rollback.append(f"pip_restore={'ok' if restored.returncode == 0 else 'failed'}")
        if n8n_updated and n8n_current:
            restored_n8n = _run(
                [NPM_BIN, "install", "--prefix", N8N_PACKAGE_ROOT, f"n8n@{n8n_current}"],
                timeout=1800,
                extra_env={"PATH": f"{NODE_BIN_DIR}:/usr/bin:/bin"},
            )
            rollback.append(f"n8n_restore={'ok' if restored_n8n.returncode == 0 else 'failed'}")
        report["status"] = "error"
        report["detail"] = str(exc)
        report["rollback"] = rollback
        report["restart_services"] = []

    report["finished_at"] = _iso_now()
    _write_json_atomic(PACKAGE_UPDATE_REPORT_PATH, report)
    return report
