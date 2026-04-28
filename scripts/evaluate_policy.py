#!/usr/bin/env python3
"""Evaluate a MahjongTransformerV2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mahjong_transformer_v2 import MahjongTransformerConfig, MahjongTransformerV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", default="1,3,5,10", help="Comma-separated k values")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.data, allow_pickle=True)
    dataset = TensorDataset(
        torch.tensor(data["static_features"], dtype=torch.float32),
        torch.tensor(data["sequence_features"], dtype=torch.float32),
        torch.tensor(data["hand_counts"], dtype=torch.float32),
        torch.tensor(data["aka_flags"], dtype=torch.float32),
        torch.tensor(data["valid_masks"], dtype=torch.float32),
        torch.tensor(data["labels"], dtype=torch.long),
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = MahjongTransformerV2(MahjongTransformerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device)
    model.eval()

    top_ks = sorted({int(k) for k in args.top_k.split(",") if k.strip()})
    total = 0
    top_correct = {k: 0 for k in top_ks}
    legal = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for static, sequence, hand_counts, aka_flags, valid_mask, labels in DataLoader(dataset, batch_size=args.batch_size):
            static = static.to(args.device)
            sequence = sequence.to(args.device)
            hand_counts = hand_counts.to(args.device)
            aka_flags = aka_flags.to(args.device)
            valid_mask = valid_mask.to(args.device)
            labels = labels.to(args.device)
            logits = model(static, sequence, hand_counts, aka_flags, valid_mask)
            pred = logits.argmax(dim=-1)
            total += labels.numel()
            for k in top_ks:
                _, pred_k = logits.topk(min(k, logits.shape[-1]), dim=-1)
                top_correct[k] += pred_k.eq(labels.unsqueeze(-1)).any(dim=-1).sum().item()
            legal += (valid_mask.gather(1, pred.unsqueeze(1)).squeeze(1) > 0).sum().item()
            loss_sum += criterion(logits, labels).item()

    result = {
        "samples": total,
        "loss": loss_sum / max(total, 1),
        "legal_rate": legal / max(total, 1),
    }
    for k in top_ks:
        result[f"top{k}"] = top_correct[k] / max(total, 1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
