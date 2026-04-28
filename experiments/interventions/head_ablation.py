"""Attention head ablation utilities."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F


@torch.no_grad()
def run_head_ablation(model, batch: Dict[str, torch.Tensor], layer: int, heads: Iterable[int]) -> Dict[str, torch.Tensor]:
    clean_logits = model(
        batch["static"],
        batch["sequence"],
        batch.get("hand_counts"),
        batch.get("aka_flags"),
        batch.get("valid_mask"),
    )
    ablated_logits = model(
        batch["static"],
        batch["sequence"],
        batch.get("hand_counts"),
        batch.get("aka_flags"),
        batch.get("valid_mask"),
        head_ablation={layer: list(heads)},
    )
    clean_prob = F.softmax(clean_logits, dim=-1)
    ablated_prob = F.softmax(ablated_logits, dim=-1)
    kl = F.kl_div(ablated_prob.log(), clean_prob, reduction="none").sum(dim=-1)
    return {
        "clean_logits": clean_logits,
        "ablated_logits": ablated_logits,
        "kl": kl,
        "clean_pred": clean_logits.argmax(dim=-1),
        "ablated_pred": ablated_logits.argmax(dim=-1),
    }


@torch.no_grad()
def head_importance_scores(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    n_layers = model.config.n_layers
    n_heads = model.config.n_heads
    scores = torch.zeros(n_layers, n_heads, device=batch["static"].device)
    for layer in range(n_layers):
        for head in range(n_heads):
            result = run_head_ablation(model, batch, layer, [head])
            scores[layer, head] = result["kl"].mean()
    return scores
