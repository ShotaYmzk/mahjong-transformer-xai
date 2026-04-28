#!/usr/bin/env python3
"""Train MahjongTransformerV2 on a leak-safe NPZ discard dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mahjong_transformer_v2 import MahjongTransformerConfig, MahjongTransformerV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_dataset(path: str) -> TensorDataset:
    data = np.load(path, allow_pickle=True)
    tensors = [
        torch.tensor(data["static_features"], dtype=torch.float32),
        torch.tensor(data["sequence_features"], dtype=torch.float32),
        torch.tensor(data["hand_counts"], dtype=torch.float32),
        torch.tensor(data["aka_flags"], dtype=torch.float32),
        torch.tensor(data["valid_masks"], dtype=torch.float32),
        torch.tensor(data["labels"], dtype=torch.long),
    ]
    return TensorDataset(*tensors)


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    _, pred = logits.topk(k, dim=-1)
    return pred.eq(labels.unsqueeze(-1)).any(dim=-1).float().mean().item()


def run_epoch(model, loader, criterion, optimizer=None, device="cpu") -> dict:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total = 0
    for static, sequence, hand_counts, aka_flags, valid_mask, labels in loader:
        static = static.to(device)
        sequence = sequence.to(device)
        hand_counts = hand_counts.to(device)
        aka_flags = aka_flags.to(device)
        valid_mask = valid_mask.to(device)
        labels = labels.to(device)
        logits = model(static, sequence, hand_counts, aka_flags, valid_mask)
        loss = criterion(logits, labels)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        batch_size = labels.shape[0]
        total += batch_size
        total_loss += loss.item() * batch_size
        total_top1 += batch_accuracy(logits, labels, 1) * batch_size
        total_top3 += batch_accuracy(logits, labels, min(3, logits.shape[-1])) * batch_size
    return {
        "loss": total_loss / max(total, 1),
        "top1": total_top1 / max(total, 1),
        "top3": total_top3 / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.data)
    if len(dataset) == 0:
        raise SystemExit("Dataset is empty")
    valid_size = max(1, int(len(dataset) * args.valid_ratio)) if len(dataset) > 10 else max(1, len(dataset) // 5)
    train_size = len(dataset) - valid_size
    train_ds, valid_ds = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    sample = dataset[0]
    config = MahjongTransformerConfig(
        static_dim=sample[0].shape[-1],
        sequence_dim=sample[1].shape[-1],
        max_sequence_length=sample[1].shape[-2],
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )
    model = MahjongTransformerV2(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, args.device)
        valid_metrics = run_epoch(model, valid_loader, criterion, None, args.device)
        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config.__dict__, "history": history}, output)


if __name__ == "__main__":
    main()
