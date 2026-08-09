#!/usr/bin/env python3
"""Verify the monthly5 volume candidate remains forward-shadow only."""

import argparse
import json
import time
from pathlib import Path

import monthly5_volume_forward_shadow as shadow


DEFAULT_STATE = Path(".runtime/data/btcusdt_monthly5_volume_forward_state.json")
DEFAULT_HISTORY = Path(".runtime/data/btcusdt_monthly5_volume_forward_history.jsonl")


def verify(state, rows, max_age_sec):
    failures = []
    if state.get("candidate_id") != shadow.CANDIDATE_ID:
        failures.append("candidate_id mismatch")
    if state.get("shadow_only") is not True:
        failures.append("state must be shadow_only")
    if state.get("execution_allowed") is not False:
        failures.append("state execution_allowed must be false")
    if not rows:
        failures.append("history is empty")
    elif any(row.get("execution_allowed") is not False for row in rows):
        failures.append("all history rows must set execution_allowed=false")
    elif any(row.get("shadow_only") is not True for row in rows):
        failures.append("all history rows must set shadow_only=true")
    age = time.time() - float(state.get("updated_ts") or 0)
    if age > max_age_sec:
        failures.append(f"state stale age={age:.1f}s")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    parser.add_argument("--max-age-sec", type=float, default=900.0)
    args = parser.parse_args()
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    rows = shadow.load_history(args.history)
    failures = verify(state, rows, args.max_age_sec)
    if failures:
        print("FAIL monthly5_volume_forward " + "; ".join(failures))
        return 1
    print(
        "PASS monthly5_volume_forward"
        f" rows={len(rows)} span_hours={state.get('span_hours')}"
        f" candidate_return_pct={state.get('candidate_paper', {}).get('return_pct')}"
        f" baseline_return_pct={state.get('baseline_paper', {}).get('return_pct')}"
        " execution_allowed=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
