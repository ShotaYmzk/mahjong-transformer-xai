#!/usr/bin/env python3
"""Optional Mjx cross-check harness.

Mjx is intentionally optional because its README notes build/API instability.
This script reports availability and leaves the self-contained extractor usable
when Mjx is not installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.observation_schema import build_dataset_rows_from_xml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit-files", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import mjx  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        rows, report = build_dataset_rows_from_xml(args.input, limit_files=args.limit_files)
        print(json.dumps({
            "mjx_available": False,
            "skip_reason": str(exc),
            "self_extractor_samples": len(rows),
            "self_extractor_report": report.to_dict(),
        }, ensure_ascii=False, indent=2))
        return

    rows, report = build_dataset_rows_from_xml(args.input, limit_files=args.limit_files)
    print(json.dumps({
        "mjx_available": True,
        "note": "Mjx import succeeded. Full replay comparison can be added for the installed API version.",
        "self_extractor_samples": len(rows),
        "self_extractor_report": report.to_dict(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
