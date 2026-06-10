#!/usr/bin/env python3
import os
import re
import signal
import shutil
import subprocess
import sys
from pathlib import Path

from runtime_paths import data_path, ensure_parent_dir


TUNNEL_URL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
PANEL_PORT = str(int(str(os.getenv("POSITION_PANEL_REALTIME_PORT", "8787")).strip() or "8787"))
PUBLIC_URL_PATH = ensure_parent_dir(data_path("panel_realtime_public_url.txt"))


def _fixed_public_url():
    return (
        str(
            os.getenv("POSITION_PANEL_PUBLIC_URL", "")
            or os.getenv("POSITION_PANEL_REALTIME_PUBLIC_URL", "")
            or ""
        )
        .strip()
        .rstrip("/")
    )


def _cleanup_public_url():
    try:
        Path(PUBLIC_URL_PATH).unlink(missing_ok=True)
    except Exception:
        pass


def _spawn_cloudflared():
    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        raise RuntimeError("cloudflared command not found")

    tunnel_name = str(
        os.getenv("POSITION_PANEL_CLOUDFLARED_TUNNEL_NAME", "")
        or os.getenv("CLOUDFLARED_TUNNEL_NAME", "")
        or ""
    ).strip()
    config_path = str(
        os.getenv("POSITION_PANEL_CLOUDFLARED_CONFIG", "")
        or os.getenv("CLOUDFLARED_CONFIG", "")
        or ""
    ).strip()

    if tunnel_name:
        cmd = [
            cloudflared,
            "tunnel",
            "--no-autoupdate",
            "--protocol",
            "quic",
        ]
        if config_path:
            cmd.extend(["--config", config_path])
        cmd.extend(["run", tunnel_name])
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    target = str(os.getenv("POSITION_PANEL_REALTIME_TUNNEL_TARGET", "") or "").strip()
    if not target:
        target = f"http://127.0.0.1:{PANEL_PORT}"
    cmd = [
        cloudflared,
        "tunnel",
        "--url",
        target,
        "--no-autoupdate",
        "--protocol",
        "quic",
        "--ha-connections",
        "1",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def main():
    fixed_url = _fixed_public_url()
    if fixed_url:
        Path(PUBLIC_URL_PATH).write_text(fixed_url + "\n", encoding="utf-8")
        print(f"PANEL_PUBLIC_URL={fixed_url}", flush=True)

    child = _spawn_cloudflared()

    def _shutdown(*_args):
        _cleanup_public_url()
        if child.poll() is None:
            try:
                child.terminate()
            except Exception:
                pass

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    current_url = ""
    try:
        assert child.stdout is not None
        for raw_line in child.stdout:
            line = str(raw_line or "")
            sys.stdout.write(line)
            sys.stdout.flush()
            match = TUNNEL_URL_RE.search(line)
            if not match:
                continue
            url = match.group(0).rstrip("/")
            if url == current_url:
                continue
            current_url = url
            Path(PUBLIC_URL_PATH).write_text(url + "\n", encoding="utf-8")
            print(f"PANEL_PUBLIC_URL={url}", flush=True)
        return child.wait()
    finally:
        _cleanup_public_url()


if __name__ == "__main__":
    raise SystemExit(main())
