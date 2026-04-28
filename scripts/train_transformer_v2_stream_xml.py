#!/usr/bin/env python3
"""Train MahjongTransformerV2 directly from Tenhou XML files.

This avoids building a giant in-memory/full-disk NPZ for year-scale corpora.
Each XML file is parsed, leak-checked, converted to mini-batches, trained, and
then released from memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.observation_schema import (  # noqa: E402
    DatasetRow,
    build_dataset_rows_from_xml,
    rows_to_npz_dict,
    validate_no_private_leakage,
)
from models.mahjong_transformer_v2 import MahjongTransformerConfig, MahjongTransformerV2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Tenhou XML directory")
    parser.add_argument("--output", required=True, help="Final checkpoint path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--log", default=None, help="JSONL training log")
    parser.add_argument("--checkpoint-every-files", type=int, default=1000)
    parser.add_argument(
        "--buffer-files",
        type=int,
        default=1,
        help="Parse this many XML files before training. Larger values improve GPU utilization.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from --output if it exists")
    parser.add_argument("--validation-offset-files", type=int, default=180000)
    parser.add_argument("--validation-limit-files", type=int, default=100)
    parser.add_argument("--metrics-csv", default=None, help="Optional CSV for paper plots")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


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


def train_rows(model, optimizer, criterion, rows: List[DatasetRow], batch_size: int, device: str) -> Dict[str, float]:
    tensors = rows_to_tensors(rows, device)
    n = int(tensors["labels"].shape[0])
    order = torch.randperm(n, device=device)
    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0
    total_top10 = 0
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        logits = model(
            tensors["static"][idx],
            tensors["sequence"][idx],
            tensors["hand_counts"][idx],
            tensors["aka_flags"][idx],
            tensors["valid_mask"][idx],
        )
        labels = tensors["labels"][idx]
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_n = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_n
        pred = logits.argmax(dim=-1)
        top3 = logits.topk(min(3, logits.shape[-1]), dim=-1).indices
        top5 = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
        top10 = logits.topk(min(10, logits.shape[-1]), dim=-1).indices
        total_top1 += int(pred.eq(labels).sum().item())
        total_top3 += int(top3.eq(labels[:, None]).any(dim=-1).sum().item())
        total_top5 += int(top5.eq(labels[:, None]).any(dim=-1).sum().item())
        total_top10 += int(top10.eq(labels[:, None]).any(dim=-1).sum().item())
    return {
        "samples": float(n),
        "loss_sum": total_loss,
        "top1_sum": float(total_top1),
        "top3_sum": float(total_top3),
        "top5_sum": float(total_top5),
        "top10_sum": float(total_top10),
    }


@torch.no_grad()
def evaluate_rows(model, criterion, rows: List[DatasetRow], batch_size: int, device: str) -> Dict[str, float]:
    tensors = rows_to_tensors(rows, device)
    n = int(tensors["labels"].shape[0])
    total_loss = 0.0
    total_legal = 0
    top_sums = {1: 0, 3: 0, 5: 0, 10: 0}
    model.eval()
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        logits = model(
            tensors["static"][sl],
            tensors["sequence"][sl],
            tensors["hand_counts"][sl],
            tensors["aka_flags"][sl],
            tensors["valid_mask"][sl],
        )
        labels = tensors["labels"][sl]
        total_loss += float(criterion(logits, labels).item())
        pred = logits.argmax(dim=-1)
        total_legal += int((tensors["valid_mask"][sl].gather(1, pred[:, None]).squeeze(1) > 0).sum().item())
        for k in top_sums:
            pred_k = logits.topk(min(k, logits.shape[-1]), dim=-1).indices
            top_sums[k] += int(pred_k.eq(labels[:, None]).any(dim=-1).sum().item())
    model.train()
    return {
        "samples": float(n),
        "loss_sum": total_loss,
        "legal_sum": float(total_legal),
        "top1_sum": float(top_sums[1]),
        "top3_sum": float(top_sums[3]),
        "top5_sum": float(top_sums[5]),
        "top10_sum": float(top_sums[10]),
    }


def add_metrics(totals: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key in totals:
        totals[key] += metrics.get(key, 0.0)


def normalize_metrics(prefix: str, totals: Dict[str, float]) -> Dict[str, float | int]:
    samples = max(totals["samples"], 1.0)
    result: Dict[str, float | int] = {
        f"{prefix}_samples": int(totals["samples"]),
        f"{prefix}_loss": totals["loss_sum"] / samples,
        f"{prefix}_top1": totals["top1_sum"] / samples,
        f"{prefix}_top3": totals["top3_sum"] / samples,
        f"{prefix}_top5": totals["top5_sum"] / samples,
        f"{prefix}_top10": totals["top10_sum"] / samples,
    }
    if "legal_sum" in totals:
        result[f"{prefix}_legal_rate"] = totals["legal_sum"] / samples
    return result


def append_metrics_csv(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_samples",
        "train_loss",
        "train_top1",
        "train_top3",
        "train_top5",
        "train_top10",
        "val_samples",
        "val_loss",
        "val_top1",
        "val_top3",
        "val_top5",
        "val_top10",
        "val_legal_rate",
        "skipped_files",
        "elapsed_sec",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def print_epoch_summary(row: Dict) -> None:
    val_part = ""
    if "val_top1" in row:
        val_part = (
            f" | val loss={row['val_loss']:.4f} "
            f"top1={row['val_top1']:.4f} top3={row['val_top3']:.4f} "
            f"top5={row['val_top5']:.4f} top10={row['val_top10']:.4f}"
        )
    print(
        f"[epoch {row['epoch']}] "
        f"train loss={row['train_loss']:.4f} "
        f"top1={row['train_top1']:.4f} top3={row['train_top3']:.4f} "
        f"top5={row['train_top5']:.4f} top10={row['train_top10']:.4f}"
        f"{val_part} | samples={row['train_samples']} elapsed={row['elapsed_sec']}s",
        flush=True,
    )


def save_checkpoint(path: Path, model, optimizer, config, epoch: int, file_index: int, history: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.__dict__,
            "epoch": epoch,
            "file_index": file_index,
            "history": history,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    xml_files = sorted(Path(args.input).glob("*.xml"))
    if args.limit_files:
        xml_files = xml_files[: args.limit_files]
    if not xml_files:
        raise SystemExit("No XML files found")

    config = MahjongTransformerConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    )
    model = MahjongTransformerV2(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    output = Path(args.output)
    history: List[Dict] = []
    start_epoch = 1

    if args.resume and output.exists():
        checkpoint = torch.load(output, map_location=args.device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    log_path = Path(args.log) if args.log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = Path(args.metrics_csv) if args.metrics_csv else None
    validation_files = xml_files[args.validation_offset_files :]
    if args.validation_limit_files:
        validation_files = validation_files[: args.validation_limit_files]

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        random.shuffle(xml_files)
        totals = {
            "samples": 0.0,
            "loss_sum": 0.0,
            "top1_sum": 0.0,
            "top3_sum": 0.0,
            "top5_sum": 0.0,
            "top10_sum": 0.0,
        }
        skipped = 0
        buffered_rows: List[DatasetRow] = []

        for file_index, xml_file in enumerate(xml_files, start=1):
            try:
                rows, report = build_dataset_rows_from_xml(xml_file)
                if not rows:
                    skipped += 1
                    continue
                buffered_rows.extend(rows)
                if len(buffered_rows) and file_index % args.buffer_files == 0:
                    metrics = train_rows(model, optimizer, criterion, buffered_rows, args.batch_size, args.device)
                    add_metrics(totals, metrics)
                    buffered_rows = []
            except Exception as exc:  # noqa: BLE001 - continue year-scale training.
                skipped += 1
                if log_path:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"type": "file_error", "file": str(xml_file), "error": str(exc)}, ensure_ascii=False) + "\n")

            if file_index % args.checkpoint_every_files == 0:
                save_checkpoint(output, model, optimizer, config, epoch - 1, file_index, history)
                progress_metrics = normalize_metrics("train", totals) if totals["samples"] > 0 else {}
                progress = {
                    "type": "progress",
                    "epoch": epoch,
                    "file_index": file_index,
                    "total_files": len(xml_files),
                    "samples": int(totals["samples"]),
                    "skipped_files": skipped,
                    "elapsed_sec": round(time.time() - epoch_start, 2),
                }
                progress.update(progress_metrics)
                progress["message"] = (
                    f"epoch {epoch} | files {file_index}/{len(xml_files)} "
                    f"({file_index / len(xml_files) * 100:.1f}%) | "
                    f"samples {int(totals['samples'])}"
                )
                if progress_metrics:
                    progress["message"] += (
                        f" | loss {progress_metrics['train_loss']:.4f}"
                        f" | top1 {progress_metrics['train_top1']:.4f}"
                        f" top3 {progress_metrics['train_top3']:.4f}"
                        f" top5 {progress_metrics['train_top5']:.4f}"
                        f" top10 {progress_metrics['train_top10']:.4f}"
                    )
                print(json.dumps(progress, ensure_ascii=False), flush=True)
                print(progress["message"], flush=True)
                if log_path:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(progress, ensure_ascii=False) + "\n")

        if buffered_rows:
            metrics = train_rows(model, optimizer, criterion, buffered_rows, args.batch_size, args.device)
            add_metrics(totals, metrics)

        val_totals = {
            "samples": 0.0,
            "loss_sum": 0.0,
            "legal_sum": 0.0,
            "top1_sum": 0.0,
            "top3_sum": 0.0,
            "top5_sum": 0.0,
            "top10_sum": 0.0,
        }
        for val_file in validation_files:
            try:
                val_rows, _ = build_dataset_rows_from_xml(val_file)
                if val_rows:
                    add_metrics(val_totals, evaluate_rows(model, criterion, val_rows, args.batch_size, args.device))
            except Exception as exc:  # noqa: BLE001
                if log_path:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {"type": "validation_file_error", "file": str(val_file), "error": str(exc)},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

        row = {
            "type": "epoch",
            "epoch": epoch,
            "skipped_files": skipped,
            "elapsed_sec": round(time.time() - epoch_start, 2),
        }
        row.update(normalize_metrics("train", totals))
        row.update(normalize_metrics("val", val_totals))
        history.append(row)
        save_checkpoint(output, model, optimizer, config, epoch, len(xml_files), history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        print_epoch_summary(row)
        if log_path:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if metrics_csv_path:
            append_metrics_csv(metrics_csv_path, row)


if __name__ == "__main__":
    main()
