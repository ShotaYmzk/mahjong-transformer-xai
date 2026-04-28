"""Faithfulness and behavior-change metrics."""

from .faithfulness import (
    aopc,
    decision_flip_rate,
    kl_divergence,
    logit_difference_delta,
    probability_drop,
)

__all__ = ["aopc", "decision_flip_rate", "kl_divergence", "logit_difference_delta", "probability_drop"]
