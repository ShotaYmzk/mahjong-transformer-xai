"""Activation patching for MahjongTransformerV2 hidden states."""

from __future__ import annotations

from typing import Dict

import torch


def symmetric_tile_replacement_hand_counts(hand_counts: torch.Tensor) -> torch.Tensor:
    """Swap manzu/pinzu/souzu suit blocks while preserving numbers.

    This is a conservative STR proxy for feature tensors. It never creates or
    uses opponent private hands because it only transforms the actor hand-count
    tensor already present in the leak-safe dataset row.
    """
    corrupted = hand_counts.clone()
    man = corrupted[:, 0:9].clone()
    pin = corrupted[:, 9:18].clone()
    sou = corrupted[:, 18:27].clone()
    corrupted[:, 0:9] = pin
    corrupted[:, 9:18] = sou
    corrupted[:, 18:27] = man
    return corrupted


@torch.no_grad()
def activation_patch_effect(
    model,
    clean_batch: Dict[str, torch.Tensor],
    corrupted_batch: Dict[str, torch.Tensor],
    *,
    target_layer: int,
    original_actions: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    clean_logits, clean_internals = model(
        clean_batch["static"],
        clean_batch["sequence"],
        clean_batch.get("hand_counts"),
        clean_batch.get("aka_flags"),
        clean_batch.get("valid_mask"),
        return_internals=True,
    )
    corrupted_logits = model(
        corrupted_batch["static"],
        corrupted_batch["sequence"],
        corrupted_batch.get("hand_counts"),
        corrupted_batch.get("aka_flags"),
        corrupted_batch.get("valid_mask"),
    )
    patched_logits = model(
        corrupted_batch["static"],
        corrupted_batch["sequence"],
        corrupted_batch.get("hand_counts"),
        corrupted_batch.get("aka_flags"),
        corrupted_batch.get("valid_mask"),
        activation_patch={target_layer: clean_internals["hidden_states"][target_layer]},
    )
    if original_actions is None:
        original_actions = clean_logits.argmax(dim=-1)
    corrupt_prob = corrupted_logits.softmax(dim=-1).gather(1, original_actions[:, None]).squeeze(1)
    patch_prob = patched_logits.softmax(dim=-1).gather(1, original_actions[:, None]).squeeze(1)
    return {
        "clean_logits": clean_logits,
        "corrupted_logits": corrupted_logits,
        "patched_logits": patched_logits,
        "indirect_effect": patch_prob - corrupt_prob,
        "original_actions": original_actions,
    }
