"""Audit monthly5 artifacts for lookahead and recursive-analysis evidence."""

import argparse
import json
import re
from pathlib import Path


NEGATIVE_SHIFT_PATTERNS = (
    re.compile(r"\.shift\(\s*-\d+"),
    re.compile(r"\.pct_change\([^)]*periods\s*=\s*-\d+"),
    re.compile(r"\.diff\(\s*-\d+"),
)


def audit_source(source_path):
    if not source_path:
        return {
            "available": False,
            "path": "",
            "negative_shift_hits": [],
            "blockers": ["strategy_source_missing"],
        }
    path = Path(source_path)
    if not path.exists():
        return {
            "available": False,
            "path": str(path),
            "negative_shift_hits": [],
            "blockers": ["strategy_source_missing"],
        }
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    hits = []
    for number, line in enumerate(lines, start=1):
        if any(pattern.search(line) for pattern in NEGATIVE_SHIFT_PATTERNS):
            hits.append({"line": number, "text": line.strip()[:200]})
    blockers = ["negative_shift_detected"] if hits else []
    return {
        "available": True,
        "path": str(path),
        "negative_shift_hits": hits,
        "blockers": blockers,
    }


def build_audit(*, strategy_source=None, prefix_replay_report=None):
    source = audit_source(strategy_source)
    prefix = {}
    if prefix_replay_report:
        try:
            prefix = json.loads(Path(prefix_replay_report).read_text(encoding="utf-8"))
        except Exception as exc:
            prefix = {"error": str(exc)}
    prefix_stable = bool(prefix.get("prefix_stable"))
    recursive_stable = bool(prefix.get("recursive_stable"))
    blockers = list(source["blockers"])
    if not prefix_replay_report:
        blockers.append("prefix_replay_missing")
    elif not prefix_stable:
        blockers.append("lookahead_prefix_mismatch")
    if not prefix_replay_report or not recursive_stable:
        blockers.append("recursive_analysis_missing_or_failed")
    return {
        "schema_version": 1,
        "method": "freqtrade_style_lookahead_recursive_gate",
        "passed": not blockers,
        "blockers": blockers,
        "source_audit": source,
        "prefix_replay": prefix,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-source")
    parser.add_argument("--prefix-replay-report")
    parser.add_argument("--output")
    args = parser.parse_args()
    audit = build_audit(
        strategy_source=args.strategy_source,
        prefix_replay_report=args.prefix_replay_report,
    )
    rendered = json.dumps(audit, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
