"""Faithfulness metrics for intervention experiments."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def decision_flip_rate(clean_logits: torch.Tensor, patched_logits: torch.Tensor) -> torch.Tensor:
    return (clean_logits.argmax(dim=-1) != patched_logits.argmax(dim=-1)).float().mean()


def kl_divergence(reference_logits: torch.Tensor, intervention_logits: torch.Tensor) -> torch.Tensor:
    reference_prob = F.softmax(reference_logits, dim=-1)
    intervention_log_prob = F.log_softmax(intervention_logits, dim=-1)
    return F.kl_div(intervention_log_prob, reference_prob, reduction="batchmean")


def probability_drop(clean_logits: torch.Tensor, patched_logits: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
    if actions is None:
        actions = clean_logits.argmax(dim=-1)
    clean_prob = clean_logits.softmax(dim=-1).gather(1, actions[:, None]).squeeze(1)
    patched_prob = patched_logits.softmax(dim=-1).gather(1, actions[:, None]).squeeze(1)
    return clean_prob - patched_prob


def logit_difference_delta(clean_logits: torch.Tensor, patched_logits: torch.Tensor) -> torch.Tensor:
    top2 = clean_logits.topk(2, dim=-1).indices
    best = top2[:, 0]
    runner_up = top2[:, 1]
    clean_diff = clean_logits.gather(1, best[:, None]).squeeze(1) - clean_logits.gather(1, runner_up[:, None]).squeeze(1)
    patched_diff = patched_logits.gather(1, best[:, None]).squeeze(1) - patched_logits.gather(1, runner_up[:, None]).squeeze(1)
    return clean_diff - patched_diff


def aopc(probability_drops: Iterable[torch.Tensor]) -> torch.Tensor:
    values = [drop.float().mean() for drop in probability_drops]
    if not values:
        return torch.tensor(0.0)
    return torch.stack(values).mean()
