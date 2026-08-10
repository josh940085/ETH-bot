#!/usr/bin/env python3
"""Verify the monthly 5% strategy is ready for live promotion."""

import argparse
import subprocess
import sys


SELECTED_CANDIDATE = "postlock_scale0.15_floor_pdaystopNone"


def _run_step(name: str, command: list[str]) -> tuple[bool, str]:
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
    if result.returncode == 0:
        return True, output
    return False, output or f"{name} failed with code {result.returncode}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-age-sec", type=float, default=900.0)
    parser.add_argument("--skip-history", action="store_true")
    parser.add_argument("--strategy-source")
    parser.add_argument("--prefix-replay-report")
    parser.add_argument("--trade-evidence")
    args = parser.parse_args(argv)

    runtime_command = [
        sys.executable,
        "verify_monthly5_runtime.py",
        "--max-age-sec",
        str(args.max_age_sec),
    ]
    if not args.skip_history:
        runtime_command.append("--require-history")

    bias_command = [sys.executable, "monthly5_bias_audit.py"]
    if args.strategy_source:
        bias_command.extend(["--strategy-source", args.strategy_source])
    if args.prefix_replay_report:
        bias_command.extend(["--prefix-replay-report", args.prefix_replay_report])

    execution_evidence_command = [
        sys.executable,
        "monthly5_batch_validator.py",
        "--candidate",
        SELECTED_CANDIDATE,
    ]
    if args.trade_evidence:
        execution_evidence_command.extend(["--trade-evidence", args.trade_evidence])

    steps = [
        ("artifact_consistency", [sys.executable, "verify_monthly5_candidate.py"]),
        ("bias", bias_command),
        ("execution_evidence", execution_evidence_command),
        ("runtime", runtime_command),
        ("account", [sys.executable, "verify_monthly5_account.py"]),
        (
            "promotion",
            [
                sys.executable,
                "verify_monthly5_readiness.py",
                "--max-age-sec",
                str(args.max_age_sec),
                "--require-promotion-ready",
            ],
        ),
        ("activation", [sys.executable, "verify_monthly5_activation.py"]),
    ]

    failures = []
    for name, command in steps:
        ok, output = _run_step(name, command)
        status = "PASS" if ok else "FAIL"
        first_line = output.splitlines()[0] if output else ""
        print(f"{status} monthly5_go_live_step={name} {first_line}".rstrip())
        if not ok:
            failures.append(name)
            for line in output.splitlines()[1:]:
                print(line)

    if failures:
        print(f"FAIL monthly5_go_live failed_steps={','.join(failures)}")
        return 1
    print("PASS monthly5_go_live promotion_ready=true live_gate=ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
