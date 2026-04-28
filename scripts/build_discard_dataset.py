#!/usr/bin/env python3
"""Build a leak-safe discard imitation dataset from Tenhou XML logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.observation_schema import (
    build_dataset_rows_from_xml,
    rows_to_npz_dict,
    save_rows_npz,
    validate_no_private_leakage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Tenhou XML file or directory")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--limit-files", type=int, default=None, help="Optional number of XML files to process")
    parser.add_argument(
        "--exclude-call-discards",
        action="store_true",
        help="Keep only draw-after-tsumo discard decisions",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, report = build_dataset_rows_from_xml(
        args.input,
        limit_files=args.limit_files,
        include_call_discards=not args.exclude_call_discards,
    )
    data = rows_to_npz_dict(rows)
    leakage_errors = validate_no_private_leakage(data)
    if leakage_errors:
        raise SystemExit("Leakage validation failed:\n" + "\n".join(leakage_errors[:20]))

    save_rows_npz(rows, args.output)
    report_dict = report.to_dict()
    report_dict["output"] = str(Path(args.output))
    report_dict["leakage_errors"] = leakage_errors

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
