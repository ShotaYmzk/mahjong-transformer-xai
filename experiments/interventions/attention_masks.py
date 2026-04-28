"""Layer-1 attention logit masking interventions."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def make_attention_patch(mode: str, *, layer: Optional[int] = None, heads=None, k: int = 1) -> Dict:
    if mode not in {"topk", "random", "bottomk", "uniform"}:
        raise ValueError(f"unknown attention patch mode: {mode}")
    return {"mode": mode, "layer": layer, "heads": heads, "k": k}


@torch.no_grad()
def run_attention_mask(
    model,
    batch: Dict[str, torch.Tensor],
    *,
    mode: str,
    layer: Optional[int] = None,
    heads=None,
    k: int = 1,
) -> Dict[str, torch.Tensor]:
    clean_logits = model(
        batch["static"],
        batch["sequence"],
        batch.get("hand_counts"),
        batch.get("aka_flags"),
        batch.get("valid_mask"),
    )
    patch = make_attention_patch(mode, layer=layer, heads=heads, k=k)
    patched_logits = model(
        batch["static"],
        batch["sequence"],
        batch.get("hand_counts"),
        batch.get("aka_flags"),
        batch.get("valid_mask"),
        attention_patch=patch,
    )
    return {
        "clean_logits": clean_logits,
        "patched_logits": patched_logits,
        "clean_pred": clean_logits.argmax(dim=-1),
        "patched_pred": patched_logits.argmax(dim=-1),
    }
