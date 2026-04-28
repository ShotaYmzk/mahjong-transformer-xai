#!/usr/bin/env python3
"""Train MahjongTransformerV2 from GPU-sized HDF5 shards."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mahjong_transformer_v2 import MahjongTransformerConfig, MahjongTransformerV2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--valid-shards", type=int, default=4)
    parser.add_argument("--log", default=None)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_shard(path: Path, device: str) -> Dict[str, torch.Tensor]:
    with h5py.File(path, "r") as f:
        return {
            "static": torch.tensor(f["static_features"][:], dtype=torch.float32, device=device),
            "sequence": torch.tensor(f["sequence_features"][:], dtype=torch.float32, device=device),
            "hand_counts": torch.tensor(f["hand_counts"][:], dtype=torch.float32, device=device),
            "aka_flags": torch.tensor(f["aka_flags"][:], dtype=torch.float32, device=device),
            "valid_mask": torch.tensor(f["valid_masks"][:], dtype=torch.float32, device=device),
            "labels": torch.tensor(f["labels"][:], dtype=torch.long, device=device),
        }


def topk_sums(logits: torch.Tensor, labels: torch.Tensor, top_ks: Iterable[int]) -> Dict[int, int]:
    return {
        k: int(logits.topk(min(k, logits.shape[-1]), dim=-1).indices.eq(labels[:, None]).any(dim=-1).sum().item())
        for k in top_ks
    }


def run_shard(model, criterion, tensors: Dict[str, torch.Tensor], batch_size: int, train: bool, optimizer=None) -> Dict[str, float]:
    n = int(tensors["labels"].shape[0])
    order = torch.randperm(n, device=tensors["labels"].device) if train else torch.arange(n, device=tensors["labels"].device)
    totals = {"samples": float(n), "loss_sum": 0.0, "legal_sum": 0.0, "top1_sum": 0.0, "top3_sum": 0.0, "top5_sum": 0.0, "top10_sum": 0.0}
    model.train(train)

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
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        batch_n = int(labels.shape[0])
        totals["loss_sum"] += float(loss.item()) * batch_n
        pred = logits.argmax(dim=-1)
        totals["legal_sum"] += float((tensors["valid_mask"][idx].gather(1, pred[:, None]).squeeze(1) > 0).sum().item())
        sums = topk_sums(logits, labels, (1, 3, 5, 10))
        totals["top1_sum"] += sums[1]
        totals["top3_sum"] += sums[3]
        totals["top5_sum"] += sums[5]
        totals["top10_sum"] += sums[10]
    return totals


def add_totals(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0.0) + value


def normalize(prefix: str, totals: Dict[str, float]) -> Dict[str, float | int]:
    samples = max(totals.get("samples", 0.0), 1.0)
    return {
        f"{prefix}_samples": int(totals.get("samples", 0.0)),
        f"{prefix}_loss": totals.get("loss_sum", 0.0) / samples,
        f"{prefix}_top1": totals.get("top1_sum", 0.0) / samples,
        f"{prefix}_top3": totals.get("top3_sum", 0.0) / samples,
        f"{prefix}_top5": totals.get("top5_sum", 0.0) / samples,
        f"{prefix}_top10": totals.get("top10_sum", 0.0) / samples,
        f"{prefix}_legal_rate": totals.get("legal_sum", 0.0) / samples,
    }


def save_checkpoint(path: Path, model, optimizer, config, epoch: int, history: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": config.__dict__, "epoch": epoch, "history": history}, path)


def append_csv(path: Path, row: Dict) -> None:
    fields = [
        "epoch", "train_samples", "train_loss", "train_top1", "train_top3", "train_top5", "train_top10", "train_legal_rate",
        "val_samples", "val_loss", "val_top1", "val_top3", "val_top5", "val_top10", "val_legal_rate", "elapsed_sec",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    shards = sorted(Path(args.shards_dir).glob("*.h5"))
    if not shards:
        raise SystemExit("No HDF5 shards found")

    valid_shards = shards[-args.valid_shards :] if args.valid_shards else []
    train_shards = shards[: -args.valid_shards] if args.valid_shards else shards
    config = MahjongTransformerConfig(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, d_ff=args.d_ff)
    model = MahjongTransformerV2(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    output = Path(args.output)
    history: List[Dict] = []
    start_epoch = 1

    if args.resume and output.exists():
        ckpt = torch.load(output, map_location=args.device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = list(ckpt.get("history", []))
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    log_path = Path(args.log) if args.log else None
    csv_path = Path(args.metrics_csv) if args.metrics_csv else None

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        random.shuffle(train_shards)
        train_totals: Dict[str, float] = {}
        val_totals: Dict[str, float] = {}

        for i, shard in enumerate(train_shards, start=1):
            tensors = load_shard(shard, args.device)
            add_totals(train_totals, run_shard(model, criterion, tensors, args.batch_size, train=True, optimizer=optimizer))
            if i % 10 == 0:
                partial = normalize("train", train_totals)
                print(
                    f"epoch {epoch} shard {i}/{len(train_shards)} | "
                    f"loss {partial['train_loss']:.4f} top1 {partial['train_top1']:.4f} "
                    f"top3 {partial['train_top3']:.4f} top5 {partial['train_top5']:.4f} top10 {partial['train_top10']:.4f}",
                    flush=True,
                )

        for shard in valid_shards:
            tensors = load_shard(shard, args.device)
            add_totals(val_totals, run_shard(model, criterion, tensors, args.batch_size, train=False))

        row = {"type": "epoch", "epoch": epoch, "elapsed_sec": round(time.time() - start, 2)}
        row.update(normalize("train", train_totals))
        row.update(normalize("val", val_totals))
        history.append(row)
        save_checkpoint(output, model, optimizer, config, epoch, history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        print(
            f"[epoch {epoch}] train loss={row['train_loss']:.4f} top1={row['train_top1']:.4f} "
            f"top3={row['train_top3']:.4f} top5={row['train_top5']:.4f} top10={row['train_top10']:.4f} | "
            f"val loss={row['val_loss']:.4f} top1={row['val_top1']:.4f} top3={row['val_top3']:.4f} "
            f"top5={row['val_top5']:.4f} top10={row['val_top10']:.4f}",
            flush=True,
        )
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if csv_path:
            append_csv(csv_path, row)


if __name__ == "__main__":
    main()
