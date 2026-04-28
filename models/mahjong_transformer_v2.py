"""Hook-friendly Mahjong Transformer for 34-kind discard prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class MahjongTransformerConfig:
    static_dim: int = 157
    sequence_dim: int = 6
    hand_dim: int = 34
    aka_dim: int = 3
    num_actions: int = 34
    max_sequence_length: int = 60
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 256
    dropout: float = 0.1


class HookedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        *,
        layer_idx: int,
        attention_patch: Optional[Dict[str, Any]] = None,
        head_ablation: Optional[Dict[int, List[int]]] = None,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        logits = apply_attention_patch(logits, layer_idx, attention_patch)
        weights = F.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        head_outputs = torch.matmul(weights, v)

        if head_ablation and layer_idx in head_ablation:
            for head in head_ablation[layer_idx]:
                if 0 <= head < self.n_heads:
                    head_outputs[:, head, :, :] = 0.0

        merged = head_outputs.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.out_proj(merged)
        return out, {
            "attn_logits": logits.detach(),
            "attn_weights": weights.detach(),
            "head_outputs": head_outputs.detach(),
        }


class HookedTransformerBlock(nn.Module):
    def __init__(self, config: MahjongTransformerConfig):
        super().__init__()
        self.attn = HookedSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        *,
        layer_idx: int,
        attention_patch: Optional[Dict[str, Any]] = None,
        head_ablation: Optional[Dict[int, List[int]]] = None,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        attn_out, internals = self.attn(
            self.norm1(x),
            layer_idx=layer_idx,
            attention_patch=attention_patch,
            head_ablation=head_ablation,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x, internals


class MahjongTransformerV2(nn.Module):
    def __init__(self, config: Optional[MahjongTransformerConfig] = None, **kwargs: Any):
        super().__init__()
        if config is None:
            config = MahjongTransformerConfig(**kwargs)
        self.config = config
        self.sequence_proj = nn.Linear(config.sequence_dim, config.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.max_sequence_length, config.d_model))
        self.blocks = nn.ModuleList([HookedTransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)

        static_input_dim = config.static_dim + config.hand_dim + config.aka_dim
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_actions),
        )

    def forward(
        self,
        static: Tensor,
        sequence: Tensor,
        hand_counts: Optional[Tensor] = None,
        aka_flags: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        *,
        return_internals: bool = False,
        attention_patch: Optional[Dict[str, Any]] = None,
        head_ablation: Optional[Dict[int, List[int]]] = None,
        activation_patch: Optional[Dict[int, Tensor]] = None,
    ) -> Tensor | tuple[Tensor, Dict[str, List[Tensor]]]:
        batch = static.shape[0]
        if hand_counts is None:
            hand_counts = static.new_zeros((batch, self.config.hand_dim))
        if aka_flags is None:
            aka_flags = static.new_zeros((batch, self.config.aka_dim))

        x = self.sequence_proj(sequence)
        x = x + self.position_embedding[:, : x.shape[1], :]

        internals: Dict[str, List[Tensor]] = {
            "hidden_states": [],
            "attn_logits": [],
            "attn_weights": [],
            "head_outputs": [],
        }
        for layer_idx, block in enumerate(self.blocks):
            x, block_internals = block(
                x,
                layer_idx=layer_idx,
                attention_patch=attention_patch,
                head_ablation=head_ablation,
            )
            if activation_patch and layer_idx in activation_patch:
                patch_value = activation_patch[layer_idx].to(dtype=x.dtype, device=x.device)
                x = patch_value if patch_value.shape == x.shape else x
            internals["hidden_states"].append(x.detach())
            for key in ("attn_logits", "attn_weights", "head_outputs"):
                internals[key].append(block_internals[key])

        seq_repr = self.final_norm(x).mean(dim=1)
        static_repr = self.static_encoder(torch.cat([static, hand_counts, aka_flags], dim=-1))
        logits = self.policy_head(torch.cat([seq_repr, static_repr], dim=-1))
        if valid_mask is not None:
            logits = logits.masked_fill(valid_mask <= 0, -1e9)
        if return_internals:
            return logits, internals
        return logits


def apply_attention_patch(logits: Tensor, layer_idx: int, patch: Optional[Dict[str, Any]]) -> Tensor:
    if not patch:
        return logits
    target_layer = patch.get("layer")
    if target_layer is not None and target_layer != layer_idx:
        return logits

    mode = patch.get("mode")
    heads = patch.get("heads")
    patched = logits.clone()
    head_slice = heads if heads is not None else range(logits.shape[1])

    if mode in {"topk", "bottomk"}:
        k = int(patch.get("k", 1))
        source = logits[:, head_slice]
        _, indices = torch.topk(source, k=k, dim=-1, largest=(mode == "topk"))
        patched[:, head_slice] = patched[:, head_slice].scatter(-1, indices, -1e9)
    elif mode == "random":
        k = int(patch.get("k", 1))
        rand = torch.rand_like(logits[:, head_slice])
        _, indices = torch.topk(rand, k=k, dim=-1)
        patched[:, head_slice] = patched[:, head_slice].scatter(-1, indices, -1e9)
    elif mode == "uniform":
        patched[:, head_slice] = 0.0
    elif mode == "indices":
        indices = patch["indices"].to(device=logits.device)
        patched[:, head_slice] = patched[:, head_slice].scatter(-1, indices, -1e9)
    return patched
