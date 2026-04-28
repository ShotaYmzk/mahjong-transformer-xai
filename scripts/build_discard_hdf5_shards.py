#!/usr/bin/env python3
"""Build GPU-sized HDF5 shards from Tenhou XML discard examples."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.observation_schema import DatasetRow, build_dataset_rows_from_xml, rows_to_npz_dict, validate_no_private_leakage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-shard", type=int, default=250_000)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--compression", default="lzf", choices=["none", "lzf", "gzip"])
    parser.add_argument("--report", default=None)
    parser.add_argument("--progress-every-files", type=int, default=1000)
    return parser.parse_args()


def write_shard(rows: List[DatasetRow], output_dir: Path, shard_index: int, compression: str) -> Dict:
    data = rows_to_npz_dict(rows)
    errors = validate_no_private_leakage(data)
    if errors:
        raise ValueError("Leakage validation failed: " + "; ".join(errors[:5]))

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"discard_shard_{shard_index:05d}.h5"
    compression_arg = None if compression == "none" else compression
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            if key == "metadata":
                encoded = np.array([json.dumps(x, ensure_ascii=False) for x in value], dtype=h5py.string_dtype("utf-8"))
                f.create_dataset(key, data=encoded, compression=compression_arg)
            else:
                f.create_dataset(key, data=value, compression=compression_arg)
        f.attrs["num_samples"] = int(data["labels"].shape[0])
        f.attrs["schema"] = "leak_safe_discard_v1"
    return {"path": str(path), "samples": int(data["labels"].shape[0])}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.xml"))
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit("No XML files found")

    output_dir = Path(args.output_dir)
    buffer: List[DatasetRow] = []
    report = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "files_total": len(files),
        "files_processed": 0,
        "samples": 0,
        "shards": [],
        "skipped_files": 0,
        "errors": [],
    }
    start = time.time()
    shard_index = 0

    for file_index, xml_file in enumerate(files, start=1):
        try:
            rows, extraction_report = build_dataset_rows_from_xml(xml_file)
            buffer.extend(rows)
            report["samples"] += len(rows)
            report["files_processed"] += 1
        except Exception as exc:  # noqa: BLE001
            report["skipped_files"] += 1
            report["errors"].append(f"{xml_file.name}: {exc}")

        while len(buffer) >= args.samples_per_shard:
            shard_rows = buffer[: args.samples_per_shard]
            buffer = buffer[args.samples_per_shard :]
            info = write_shard(shard_rows, output_dir, shard_index, args.compression)
            report["shards"].append(info)
            shard_index += 1
            print(json.dumps({"type": "shard", **info}, ensure_ascii=False), flush=True)

        if file_index % args.progress_every_files == 0:
            progress = {
                "type": "progress",
                "file_index": file_index,
                "total_files": len(files),
                "samples": report["samples"],
                "shards": len(report["shards"]),
                "buffered_samples": len(buffer),
                "elapsed_sec": round(time.time() - start, 2),
            }
            print(json.dumps(progress, ensure_ascii=False), flush=True)

    if buffer:
        info = write_shard(buffer, output_dir, shard_index, args.compression)
        report["shards"].append(info)
        print(json.dumps({"type": "shard", **info}, ensure_ascii=False), flush=True)

    report["elapsed_sec"] = round(time.time() - start, 2)
    report["num_shards"] = len(report["shards"])
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"type": "complete", **report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
