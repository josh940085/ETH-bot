"""Report isolated research-tool versions without importing them into the live bot."""

import argparse
import json
import subprocess
from pathlib import Path


TOOLS = {
    "freqtrade": {
        "python": ".runtime/research_envs/freqtrade/bin/python",
        "package": "freqtrade",
        "expected": "2026.7",
    },
    "hftbacktest": {
        "python": ".runtime/research_envs/hftbacktest/bin/python",
        "package": "hftbacktest",
        "expected": "2.4.4",
    },
    "cryptofeed": {
        "python": ".runtime/research_envs/cryptofeed/bin/python",
        "package": "cryptofeed",
        "expected": "2.4.1",
    },
    "nautilus_trader": {
        "python": ".runtime/research_envs/nautilus/bin/python",
        "package": "nautilus_trader",
        "expected": "1.227.0",
    },
    "river": {
        "python": ".runtime/research_envs/river/bin/python",
        "package": "river",
        "expected": "0.25.0",
    },
}


def _probe_package(root, config):
    python = root / config["python"]
    if not python.exists():
        return {"installed": False, "version": "", "expected": config["expected"]}
    script = (
        "import importlib.metadata as m,platform;"
        f"print(m.version('{config['package']}'));"
        "print(platform.python_version())"
    )
    result = subprocess.run(
        [str(python), "-c", script],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    lines = result.stdout.strip().splitlines()
    version = lines[0] if lines else ""
    python_version = lines[1] if len(lines) > 1 else ""
    python_supported = python_version.startswith("3.12.")
    return {
        "installed": (
            result.returncode == 0
            and version == config["expected"]
            and python_supported
        ),
        "version": version,
        "expected": config["expected"],
        "python_version": python_version,
        "python_supported": python_supported,
        "error": result.stderr.strip() if result.returncode else "",
    }


def build_status(root=Path(".")):
    root = Path(root).resolve()
    tools = {name: _probe_package(root, config) for name, config in TOOLS.items()}
    tools["binance_public_data"] = {
        "installed": (root / "binance_trade_history.py").exists(),
        "mode": "official_archive_checksum_and_tick_replay",
    }
    tools["freqtrade_validation"] = {
        "installed": (root / "archive" / "monthly5" / "monthly5_bias_audit.py").exists(),
        "mode": "local_lookahead_and_recursive_gate",
        "full_freqtrade_runtime_installed": tools["freqtrade"]["installed"],
    }
    tools["market_regime_drift"] = {
        "installed": (root / "market_regime_drift.py").exists(),
        "mode": "river_adwin_shadow_only",
        "live_control_enabled": False,
    }
    return {
        "schema_version": 1,
        "ready": all(item.get("installed") for item in tools.values()),
        "tools": tools,
        "excluded": {
            "tardis": {
                "installed": False,
                "reason": "explicitly_excluded_by_user",
            }
        },
        "research_dependencies_isolated_from_live": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    status = build_status()
    rendered = json.dumps(status, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if status["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
