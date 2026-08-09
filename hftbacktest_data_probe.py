"""Check whether captured microstructure data is sufficient for hftbacktest."""

import argparse
import importlib.metadata
import json
from pathlib import Path


def probe(path):
    source = Path(path)
    counts = {"trade": 0, "l2_book": 0}
    malformed = 0
    if source.exists():
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                row_type = str(row.get("type") or "")
                if row_type in counts:
                    counts[row_type] += 1
    blockers = []
    if not source.exists():
        blockers.append("microstructure_capture_missing")
    if counts["trade"] <= 0:
        blockers.append("trade_ticks_missing")
    if counts["l2_book"] <= 0:
        blockers.append("l2_book_updates_missing")
    if malformed:
        blockers.append("malformed_capture_rows")
    try:
        version = importlib.metadata.version("hftbacktest")
    except importlib.metadata.PackageNotFoundError:
        version = ""
        blockers.append("hftbacktest_not_installed")
    return {
        "schema_version": 1,
        "ready": not blockers,
        "path": str(source),
        "hftbacktest_version": version,
        "counts": counts,
        "malformed_rows": malformed,
        "blockers": blockers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = probe(args.input)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
