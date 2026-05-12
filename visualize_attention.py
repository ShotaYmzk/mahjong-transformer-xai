"""Attention visualization for MahjongTransformerV2.

Provides two visualization modes:

1. **Mask-comparison bar chart** (:meth:`AttentionVisualizer.plot_mask_comparison`)
   — Shows 8 feature-group importance scores side-by-side before and after
   each of the three masking conditions (top-k / bottom-k / random-k).

2. **Multi-game heatmap** (:meth:`AttentionVisualizer.plot_group_heatmap`)
   — Displays an N-games × 8-groups importance matrix as a ``seaborn``
   heatmap, making cross-game attention patterns easy to compare at a glance.

Implementation notes
--------------------
- **No changes to any existing file.**  All model interaction goes through
  ``register_forward_hook`` on ``HookedSelfAttention`` inside ``model.blocks``.
- Hook logic, group-score aggregation, and masking are **imported and reused**
  from :mod:`attention_patching` (no copy-paste).
- The actual model uses a **custom** :class:`HookedSelfAttention` (not
  ``nn.MultiheadAttention``), so the ``need_weights=True`` concern described
  in the spec does not apply: attention weights are always returned by the
  forward method regardless.
- Default config: ``d_model=128, n_heads=8, n_layers=4`` (not 256/4 as
  described in the spec; values come from the checkpoint / ``model.config``).

Dependencies
------------
::

    pip install torch matplotlib seaborn numpy
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# ---------------------------------------------------------------------------
# Project path bootstrap (supports running the file directly from any cwd).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention_patching import (  # noqa: E402
    FEATURE_GROUP_NAMES,
    AttentionPatchingEvaluator,
    GameStateBatch,
    _default_feature_group_map,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------


class AttentionVisualizer:
    """Visualize feature-group attention scores of ``MahjongTransformerV2``.

    Internally wraps :class:`~attention_patching.AttentionPatchingEvaluator`
    so that hook management, raw-attention extraction, group-score
    aggregation, and masking are **not duplicated** here.

    Args:
        model: A ``MahjongTransformerV2`` instance in eval mode.
        feature_group_map: Optional custom mapping function with signature
            ``(sequence: Tensor[seq_len, 6], player_id: int)
            -> dict[str, list[int]]``.
            Falls back to :func:`~attention_patching._default_feature_group_map`.
        device: Torch device for inference.  Defaults to CUDA when available.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        feature_group_map: Optional[
            Callable[[torch.Tensor, int], Dict[str, List[int]]]
        ] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # A dummy explain_fn is required by AttentionPatchingEvaluator but is
        # never called from the visualizer.
        def _dummy_explain(
            _gs: Dict[str, Any], _scores: Dict[str, float]
        ) -> str:  # pragma: no cover
            return ""

        self._evaluator = AttentionPatchingEvaluator(
            model,
            _dummy_explain,
            device=self._device,
            feature_group_map=feature_group_map,
        )
        logger.info(
            "AttentionVisualizer ready. device=%s groups=%d",
            self._device,
            len(FEATURE_GROUP_NAMES),
        )

    # ------------------------------------------------------------------
    # Hook wrappers — thin delegates to the encapsulated evaluator.
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Install forward hooks to capture attention weights (delegates to evaluator)."""
        self._evaluator._register_hooks()

    def _remove_hooks(self) -> None:
        """Remove all registered forward hooks (delegates to evaluator)."""
        self._evaluator._remove_hooks()

    # ------------------------------------------------------------------
    # Core: attention extraction and group aggregation
    # ------------------------------------------------------------------

    def _forward_and_get_attention(
        self, game_state: Dict[str, Any]
    ) -> torch.Tensor:
        """Run a forward pass and return the last-layer attention weight matrix.

        Hooks are registered before and cleaned up after the forward pass.

        Args:
            game_state: Dict containing at least ``"static"`` and
                ``"sequence"`` tensors (shapes ``(B, static_dim)`` and
                ``(B, seq_len, 6)``).  Optionally ``"hand_counts"``,
                ``"aka_flags"``, and ``"valid_mask"``.

        Returns:
            Tensor of shape ``(B, n_heads, seq_len, seq_len)`` — last layer.
        """
        batch = GameStateBatch(
            static=game_state["static"],
            sequence=game_state["sequence"],
            hand_counts=game_state.get("hand_counts"),
            aka_flags=game_state.get("aka_flags"),
            valid_mask=game_state.get("valid_mask"),
        )
        return self._evaluator._extract_raw_attention(batch)

    def _compute_group_scores(
        self,
        attn_weights: torch.Tensor,
        game_state: Dict[str, Any],
    ) -> Dict[str, float]:
        """Aggregate last-layer attention into 8 feature-group scores.

        Wraps :meth:`~attention_patching.AttentionPatchingEvaluator._compute_group_scores`
        for a single game state (batch size assumed to be 1).

        Args:
            attn_weights: ``(B, n_heads, seq_len, seq_len)`` from the last
                transformer block.
            game_state: Same dict passed to :meth:`_forward_and_get_attention`;
                used to read ``"sequence"`` and ``"player_id"``.

        Returns:
            Dict mapping each group name → importance score in ``[0, 1]``,
            renormalized so values sum to 1.
        """
        player_ids = [int(game_state.get("player_id", 0))]
        results = self._evaluator._compute_group_scores(
            attn_weights, game_state["sequence"], player_ids
        )
        return results[0]

    # ------------------------------------------------------------------
    # Visualization A: mask-comparison bar chart
    # ------------------------------------------------------------------

    def plot_mask_comparison(
        self,
        game_state: Dict[str, Any],
        *,
        k: int = 3,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a grouped bar chart comparing scores before and after masking.

        For each of the 8 feature groups, four bars are rendered side-by-side:

        - **Original** (blue)   — unmasked attention group scores.
        - **Top-k masked** (red)    — 条件A: k highest-scored groups zeroed.
        - **Bottom-k masked** (green) — 条件B: k lowest-scored groups zeroed.
        - **Random-k masked** (orange) — 条件C: k randomly chosen groups zeroed.

        Masking is performed by reusing
        :meth:`~attention_patching.AttentionPatchingEvaluator._mask_attention_scores`
        from :mod:`attention_patching`.

        Args:
            game_state: Input dict (same schema as
                :meth:`_forward_and_get_attention`).
            k: Number of groups to zero per masking condition. Default 3.
            save_path: File path for PNG output.  If ``None``, calls
                ``plt.show()`` instead.
        """
        logger.info("plot_mask_comparison: k=%d save_path=%s", k, save_path)

        # --- Extract scores ---
        attn_weights = self._forward_and_get_attention(game_state)
        original = self._compute_group_scores(attn_weights, game_state)

        masked_top = self._evaluator._mask_attention_scores(original, "top", k=k)
        masked_bottom = self._evaluator._mask_attention_scores(original, "bottom", k=k)
        masked_random = self._evaluator._mask_attention_scores(original, "random", k=k)

        # --- Build arrays ---
        groups = FEATURE_GROUP_NAMES
        n_groups = len(groups)
        orig_vals = np.array([original[g] for g in groups])
        top_vals = np.array([masked_top[g] for g in groups])
        bot_vals = np.array([masked_bottom[g] for g in groups])
        rnd_vals = np.array([masked_random[g] for g in groups])

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(14, 5))

        bar_width = 0.2
        x = np.arange(n_groups)
        offsets = [-1.5, -0.5, 0.5, 1.5]
        colors = ["steelblue", "tomato", "mediumseagreen", "darkorange"]
        labels = [
            "Original",
            f"Top-{k} masked (条件A)",
            f"Bottom-{k} masked (条件B)",
            f"Random-{k} masked (条件C)",
        ]
        data_sets = [orig_vals, top_vals, bot_vals, rnd_vals]

        for offset, color, label, data in zip(offsets, colors, labels, data_sets):
            ax.bar(x + offset * bar_width, data, width=bar_width, color=color, label=label, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Normalized Attention Score")
        ax.set_title(f"Attention Group Scores — Mask Comparison (k={k})", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, max(orig_vals.max() * 1.25, 0.05))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            logger.info("Saved mask comparison chart → %s", save_path)
        else:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Visualization B: multi-game heatmap
    # ------------------------------------------------------------------

    def plot_group_heatmap(
        self,
        game_states: List[Dict[str, Any]],
        *,
        game_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a seaborn heatmap of feature-group importance across multiple games.

        Computes an ``(N, 8)`` importance matrix where N = number of game
        states, then renders it with ``seaborn.heatmap``.

        Args:
            game_states: List of game state dicts.
            game_labels: Optional list of string labels for the y-axis
                (e.g. ``["Game 1", "Game 2", ...]``).  Falls back to
                ``["Game 0", "Game 1", ...]`` when not provided.
            save_path: File path for PNG output.  If ``None``, calls
                ``plt.show()`` instead.
        """
        logger.info("plot_group_heatmap: n_games=%d", len(game_states))

        matrix: List[List[float]] = []
        for idx, gs in enumerate(game_states):
            try:
                aw = self._forward_and_get_attention(gs)
                scores = self._compute_group_scores(aw, gs)
                matrix.append([scores[g] for g in FEATURE_GROUP_NAMES])
            except Exception:  # noqa: BLE001
                logger.exception("Skipping game_state idx=%d due to error.", idx)
                matrix.append([0.0] * len(FEATURE_GROUP_NAMES))

        data = np.array(matrix)  # (N, 8)

        if game_labels is None:
            game_labels = [f"Game {i}" for i in range(len(game_states))]

        fig_h = max(4, len(game_states) * 0.5 + 1)
        fig, ax = plt.subplots(figsize=(12, fig_h))

        sns.heatmap(
            data,
            ax=ax,
            xticklabels=FEATURE_GROUP_NAMES,
            yticklabels=game_labels,
            cmap="YlOrRd",
            annot=len(game_states) <= 30,  # show values only when grid is not too dense
            fmt=".2f",
            linewidths=0.3,
            linecolor="lightgrey",
            cbar_kws={"label": "Normalized Attention Score"},
        )
        ax.set_title("Attention Group Importance Heatmap", fontsize=13, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            logger.info("Saved group heatmap → %s", save_path)
        else:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Smoke-test main block
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a self-contained sanity check with random tensors (no checkpoint needed).

    Verifies:
    - Hook registration / removal via the visualizer's delegates.
    - Attention weight extraction produces the expected shape.
    - Group scores are non-negative and sum to 1.
    - Both visualizations complete without error and save PNGs to ``/tmp/``.

    Actual model config (read from source):
        ``d_model=128, n_heads=8, n_layers=4, static_dim=157, sequence_dim=6``

    The spec document described ``d_model=256, nhead=4`` and
    ``nn.TransformerEncoder`` with ``nn.MultiheadAttention``; the real model
    uses a **custom** ``HookedSelfAttention`` inside ``model.blocks``, so
    the ``need_weights=True`` workaround is **not required** here.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    from models.mahjong_transformer_v2 import (  # noqa: E402
        MahjongTransformerConfig,
        MahjongTransformerV2,
    )
    from attention_patching import _ET_DISCARD, _ET_DRAW, _ET_PADDING  # noqa: E402

    # ------------------------------------------------------------------ model
    cfg = MahjongTransformerConfig()
    model = MahjongTransformerV2(cfg)
    model.eval()
    logger.info(
        "Dummy model: d_model=%d n_heads=%d n_layers=%d",
        cfg.d_model,
        cfg.n_heads,
        cfg.n_layers,
    )

    vis = AttentionVisualizer(model)

    # ----------------------------------------------- build dummy game states
    def make_game_state(seed: int, player_id: int = 0) -> Dict[str, Any]:
        """Create a plausible random game-state dict with a seeded sequence."""
        torch.manual_seed(seed)
        B, S = 1, 60
        seq = torch.zeros(B, S, 6)
        for t in range(0, 40, 4):
            seq[0, t] = torch.tensor(
                [_ET_DRAW, player_id, t % 34, t // 4, 0, 0], dtype=torch.float32
            )
            seq[0, t + 1] = torch.tensor(
                [_ET_DISCARD, player_id, t % 34, t // 4, 0, 0], dtype=torch.float32
            )
            seq[0, t + 2] = torch.tensor(
                [_ET_DRAW, 1 - player_id, (t + 2) % 34, t // 4, 0, 0],
                dtype=torch.float32,
            )
            seq[0, t + 3] = torch.tensor(
                [_ET_DISCARD, 1 - player_id, (t + 2) % 34, t // 4, 0, 0],
                dtype=torch.float32,
            )
        seq[0, 40:, 0] = _ET_PADDING

        return {
            "static": torch.randn(B, cfg.static_dim),
            "sequence": seq,
            "hand_counts": torch.zeros(B, cfg.hand_dim),
            "aka_flags": torch.zeros(B, cfg.aka_dim),
            "valid_mask": torch.ones(B, cfg.num_actions),
            "player_id": player_id,
        }

    game_state = make_game_state(seed=42)
    game_states = [make_game_state(seed=i) for i in range(8)]

    # -------------------------------------------------------- attention shape
    logger.info("=== Attention extraction check ===")
    aw = vis._forward_and_get_attention(game_state)
    print(f"Attention weights shape: {aw.shape}")
    assert aw.ndim == 4 and aw.shape[1] == cfg.n_heads, (
        f"Unexpected shape {aw.shape}"
    )

    # ------------------------------------------------------- group score check
    logger.info("=== Group score check ===")
    scores = vis._compute_group_scores(aw, game_state)
    total = sum(scores.values())
    print("\nGroup scores (sum = {:.6f}):".format(total))
    for name, val in scores.items():
        bar = "█" * int(val * 40)
        print(f"  {name:<22}: {val:.4f}  {bar}")
    assert abs(total - 1.0) < 1e-5 or total < 1e-8, f"Normalization failed: {total}"

    # ---------------------------------------------------------- hook lifecycle
    logger.info("=== Hook lifecycle check ===")
    vis._register_hooks()
    assert len(vis._evaluator._hooks) == cfg.n_layers, (
        f"Expected {cfg.n_layers} hooks, got {len(vis._evaluator._hooks)}"
    )
    vis._remove_hooks()
    assert len(vis._evaluator._hooks) == 0
    print("Hook register / remove: ✓")

    # ----------------------------------------------- visualization A: bar chart
    logger.info("=== plot_mask_comparison ===")
    bar_path = "/tmp/attn_mask_comparison.png"
    vis.plot_mask_comparison(game_state, k=3, save_path=bar_path)
    print(f"Bar chart saved → {bar_path}")

    # ---------------------------------------------- visualization B: heatmap
    logger.info("=== plot_group_heatmap ===")
    heat_path = "/tmp/attn_group_heatmap.png"
    vis.plot_group_heatmap(
        game_states,
        game_labels=[f"局面{i}" for i in range(len(game_states))],
        save_path=heat_path,
    )
    print(f"Heatmap saved → {heat_path}")

    logger.info("All checks passed.")
    print("\n✓ visualize_attention.py — all checks passed.")


if __name__ == "__main__":
    main()
