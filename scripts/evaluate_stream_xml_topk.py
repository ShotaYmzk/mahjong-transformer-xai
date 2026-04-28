#!/usr/bin/env python3
"""Evaluate checkpoint Top-k accuracy directly on Tenhou XML files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.observation_schema import DatasetRow, build_dataset_rows_from_xml, rows_to_npz_dict, validate_no_private_leakage  # noqa: E402
from models.mahjong_transformer_v2 import MahjongTransformerConfig, MahjongTransformerV2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Tenhou XML file or directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit-files", type=int, default=100)
    parser.add_argument("--offset-files", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--top-k", default="1,3,5,10")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def list_eval_files(path: Path, offset: int, limit: int | None) -> List[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.xml"))
    if offset:
        files = files[offset:]
    if limit:
        files = files[:limit]
    return files


def rows_to_tensors(rows: List[DatasetRow], device: str) -> Dict[str, torch.Tensor]:
    data = rows_to_npz_dict(rows)
    errors = validate_no_private_leakage(data)
    if errors:
        raise ValueError("Leakage validation failed: " + "; ".join(errors[:5]))
    return {
        "static": torch.tensor(data["static_features"], dtype=torch.float32, device=device),
        "sequence": torch.tensor(data["sequence_features"], dtype=torch.float32, device=device),
        "hand_counts": torch.tensor(data["hand_counts"], dtype=torch.float32, device=device),
        "aka_flags": torch.tensor(data["aka_flags"], dtype=torch.float32, device=device),
        "valid_mask": torch.tensor(data["valid_masks"], dtype=torch.float32, device=device),
        "labels": torch.tensor(data["labels"], dtype=torch.long, device=device),
    }


@torch.no_grad()
def eval_rows(model, rows: List[DatasetRow], batch_size: int, top_ks: Iterable[int], device: str) -> Dict[str, float]:
    tensors = rows_to_tensors(rows, device)
    labels_all = tensors["labels"]
    total = int(labels_all.shape[0])
    top_correct = {k: 0 for k in top_ks}
    legal = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    for start in range(0, total, batch_size):
        sl = slice(start, min(start + batch_size, total))
        logits = model(
            tensors["static"][sl],
            tensors["sequence"][sl],
            tensors["hand_counts"][sl],
            tensors["aka_flags"][sl],
            tensors["valid_mask"][sl],
        )
        labels = labels_all[sl]
        pred = logits.argmax(dim=-1)
        loss_sum += float(criterion(logits, labels).item())
        legal += int((tensors["valid_mask"][sl].gather(1, pred[:, None]).squeeze(1) > 0).sum().item())
        for k in top_ks:
            pred_k = logits.topk(min(k, logits.shape[-1]), dim=-1).indices
            top_correct[k] += int(pred_k.eq(labels[:, None]).any(dim=-1).sum().item())

    result = {
        "samples": float(total),
        "loss_sum": loss_sum,
        "legal_sum": float(legal),
    }
    for k in top_ks:
        result[f"top{k}_sum"] = float(top_correct[k])
    return result


def main() -> None:
    args = parse_args()
    top_ks = sorted({int(k) for k in args.top_k.split(",") if k.strip()})
    files = list_eval_files(Path(args.input), args.offset_files, args.limit_files)
    if not files:
        raise SystemExit("No XML files selected for validation")

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = MahjongTransformerV2(MahjongTransformerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device)
    model.eval()

    totals = {"samples": 0.0, "loss_sum": 0.0, "legal_sum": 0.0}
    for k in top_ks:
        totals[f"top{k}_sum"] = 0.0
    skipped_files = 0
    errors: List[str] = []

    for xml_file in files:
        try:
            rows, _ = build_dataset_rows_from_xml(xml_file)
            if not rows:
                skipped_files += 1
                continue
            metrics = eval_rows(model, rows, args.batch_size, top_ks, args.device)
            for key in totals:
                totals[key] += metrics[key]
        except Exception as exc:  # noqa: BLE001
            skipped_files += 1
            errors.append(f"{xml_file.name}: {exc}")

    samples = max(totals["samples"], 1.0)
    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "files": len(files),
        "offset_files": args.offset_files,
        "samples": int(totals["samples"]),
        "loss": totals["loss_sum"] / samples,
        "legal_rate": totals["legal_sum"] / samples,
        "skipped_files": skipped_files,
        "errors": errors[:20],
    }
    for k in top_ks:
        result[f"top{k}"] = totals[f"top{k}_sum"] / samples

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
