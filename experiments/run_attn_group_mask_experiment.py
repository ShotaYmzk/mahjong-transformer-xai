#!/usr/bin/env python3
"""Attention-group masking experiment using real Tenhou XML game logs.

Measures how much the model's output distribution changes when attention
to different feature groups is suppressed.  No LLM integration required.

Masking conditions
------------------
- top-k    (条件A): zero attention to the k highest-importance groups.
- bottom-k (条件B): zero attention to the k lowest-importance groups.
- random-k (条件C): zero attention to k randomly selected groups.

Masking is applied to ALL transformer layers by injecting -1e9 into the
attention logits for key-sequence positions that belong to the masked groups.
This reuses ``apply_attention_patch`` from ``models/mahjong_transformer_v2.py``
without modifying any existing file.

Metrics (from ``experiments/metrics/faithfulness.py``)
-------------------------------------------------------
- KL divergence from baseline output distribution.
- Decision flip rate  (does the top-1 prediction change?).
- Probability drop    (how much does the top-1 probability fall?).

Outputs
---------
- Figures under project ``figure/``:
- ``figure/attn_mask_kl_comparison.png``:  bar chart, mean KL by
  masking condition and feature group.
- ``figure/attn_group_heatmap.png``:        seaborn heatmap,
  per-sample baseline group importance scores.
- ``figure/attn_mask_group_breakdown.png``:   top-k mask group frequency.
- ``outputs/results/attn_group_mask_results.csv``:   full per-sample results.
- ``outputs/results/attn_group_mask_summary.json``:  aggregated statistics.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup (run from any directory)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data.observation_schema import build_dataset_rows_from_xml, DatasetRow  # noqa: E402
from models.mahjong_transformer_v2 import MahjongTransformerConfig, MahjongTransformerV2  # noqa: E402
from experiments.metrics.faithfulness import (  # noqa: E402
    decision_flip_rate,
    kl_divergence,
    probability_drop,
)
from attention_patching import (  # noqa: E402
    FEATURE_GROUP_NAMES,
    _default_feature_group_map,
)
from visualize_attention import AttentionVisualizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
XML_DIR = Path("/home/ubuntu/Documents/tenhou_xml_2023")
CHECKPOINT = _ROOT / "outputs/impl1/hdf5_10epoch.pt"
FIGURES_DIR = _ROOT / "figure"
RESULTS_DIR = _ROOT / "outputs/results"

N_XML_FILES = 5      # number of XML files to parse
MAX_SAMPLES = 150    # maximum game states to evaluate
K_MASK = 3           # number of feature groups to mask
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helper: load model
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> MahjongTransformerV2:
    """Load a ``MahjongTransformerV2`` checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device: Target device.

    Returns:
        Model in eval mode on the requested device.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = MahjongTransformerConfig(**ckpt["config"])
    model = MahjongTransformerV2(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    logger.info(
        "Loaded checkpoint %s  (epoch=%s  d_model=%d  n_heads=%d  n_layers=%d)",
        checkpoint_path.name,
        ckpt.get("epoch", "?"),
        cfg.d_model,
        cfg.n_heads,
        cfg.n_layers,
    )
    return model


# ---------------------------------------------------------------------------
# Helper: DatasetRow → game_state dict
# ---------------------------------------------------------------------------

def row_to_game_state(row: DatasetRow, device: torch.device) -> Dict[str, Any]:
    """Convert a ``DatasetRow`` to a game-state dict for model inference.

    Args:
        row: Parsed dataset row from XML.
        device: Target device for tensors.

    Returns:
        Dict with keys ``static``, ``sequence``, ``hand_counts``, ``aka_flags``,
        ``valid_mask``, ``player_id``, and ``label``.
    """
    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)

    return {
        "static": _t(row.static_features),
        "sequence": _t(row.sequence_features),
        "hand_counts": _t(row.hand_counts),
        "aka_flags": _t(row.aka_flags),
        "valid_mask": _t(row.valid_mask),
        "player_id": int(row.metadata.get("player_id", 0)),
        "label": int(row.label),
    }


# ---------------------------------------------------------------------------
# Helper: build attention-patch indices for masked key positions
# ---------------------------------------------------------------------------

def build_position_patch(
    positions: List[int],
    B: int,
    n_heads: int,
    seq_len: int,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Create an ``attention_patch`` dict that suppresses attention to given positions.

    Sets attention logits for the specified key-sequence positions to ``-1e9``
    in ALL transformer layers, effectively zeroing them in the softmax weights.

    Args:
        positions: Key-sequence positions to mask (0-based, values < seq_len).
        B: Batch size.
        n_heads: Number of attention heads.
        seq_len: Sequence length (key dimension).
        device: Device for the indices tensor.

    Returns:
        ``attention_patch`` dict, or ``None`` if ``positions`` is empty.
    """
    if not positions:
        return None
    K = len(positions)
    idx = torch.tensor(positions, dtype=torch.long, device=device)
    # Expand to (B, n_heads, seq_len, K) — mask same key positions for all
    # queries and all heads.
    indices = idx.view(1, 1, 1, K).expand(B, n_heads, seq_len, K).contiguous()
    return {"mode": "indices", "indices": indices}


# ---------------------------------------------------------------------------
# Helper: select groups to mask based on condition
# ---------------------------------------------------------------------------

def select_groups(
    group_scores: Dict[str, float],
    condition: str,
    k: int,
) -> List[str]:
    """Return the names of k feature groups selected by the masking condition.

    Args:
        group_scores: Dict of group name → importance score.
        condition: ``'top'``, ``'bottom'``, or ``'random'``.
        k: Number of groups to select.

    Returns:
        List of k group names.

    Raises:
        ValueError: For unknown condition strings.
    """
    names = list(group_scores)
    values = [group_scores[n] for n in names]

    if condition == "top":
        order = sorted(range(len(names)), key=lambda i: values[i], reverse=True)
    elif condition == "bottom":
        order = sorted(range(len(names)), key=lambda i: values[i])
    elif condition == "random":
        order = random.sample(range(len(names)), len(names))
    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    return [names[i] for i in order[:k]]


# ---------------------------------------------------------------------------
# Core per-sample experiment
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sample(
    model: MahjongTransformerV2,
    game_state: Dict[str, Any],
    *,
    k: int = 3,
    device: torch.device,
) -> Dict[str, Any]:
    """Run baseline + three masking conditions for one game state.

    Args:
        model: Loaded ``MahjongTransformerV2``.
        game_state: Dict from :func:`row_to_game_state`.
        k: Number of groups to mask.
        device: Inference device.

    Returns:
        Dict with keys:
        - ``group_scores``: baseline normalized importance per group.
        - ``condition_results``: per-condition dict with
          ``kl``, ``flip``, ``prob_drop``, ``masked_groups``.
    """
    cfg = model.config
    seq = game_state["sequence"]   # (1, S, 6)
    pid = game_state["player_id"]

    # --- Baseline forward ---
    logits_base, internals = model(
        game_state["static"],
        seq,
        game_state["hand_counts"],
        game_state["aka_flags"],
        game_state["valid_mask"],
        return_internals=True,
    )

    # --- Attention group scores (last layer, mean over heads + query positions) ---
    last_attn = internals["attn_weights"][-1]          # (1, H, S, S)
    importance = last_attn.mean(dim=(1, 2)).squeeze(0)  # (S,)
    group_positions = _default_feature_group_map(seq[0].cpu(), pid)

    raw_scores: Dict[str, float] = {}
    for name in FEATURE_GROUP_NAMES:
        pos = group_positions.get(name, [])
        raw_scores[name] = float(importance[pos].sum().item()) if pos else 0.0

    total = sum(raw_scores.values())
    group_scores = {n: v / total for n, v in raw_scores.items()} if total > 1e-8 else raw_scores

    # --- Three masking conditions ---
    condition_results: Dict[str, Dict[str, Any]] = {}

    for cond in ("top", "bottom", "random"):
        groups_to_mask = select_groups(group_scores, cond, k=k)
        positions_to_mask: List[int] = []
        for g in groups_to_mask:
            positions_to_mask.extend(group_positions.get(g, []))
        positions_to_mask = sorted(set(positions_to_mask))

        patch = build_position_patch(
            positions_to_mask,
            B=1,
            n_heads=cfg.n_heads,
            seq_len=seq.shape[1],
            device=device,
        )

        if patch is not None:
            logits_masked = model(
                game_state["static"],
                seq,
                game_state["hand_counts"],
                game_state["aka_flags"],
                game_state["valid_mask"],
                attention_patch=patch,
            )
        else:
            logits_masked = logits_base.clone()

        kl = float(kl_divergence(logits_base, logits_masked).item())
        flip = float(decision_flip_rate(logits_base, logits_masked).item())
        pdrop = float(probability_drop(logits_base, logits_masked).item())

        condition_results[cond] = {
            "kl": kl,
            "flip": flip,
            "prob_drop": pdrop,
            "masked_groups": groups_to_mask,
            "n_positions_masked": len(positions_to_mask),
        }

    return {
        "group_scores": group_scores,
        "condition_results": condition_results,
        "label": game_state["label"],
        "player_id": pid,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_kl_comparison(df: pd.DataFrame, save_path: Path) -> None:
    """Bar chart: mean KL divergence by masking condition.

    Also overlays individual sample values as scatter points.

    Args:
        df: DataFrame with columns ``condition`` and ``kl``.
        save_path: Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    cond_labels = {
        "top": f"Top-{K_MASK} masked\n(条件A)",
        "bottom": f"Bottom-{K_MASK} masked\n(条件B)",
        "random": f"Random-{K_MASK} masked\n(条件C)",
    }
    colors = {"top": "tomato", "bottom": "mediumseagreen", "random": "darkorange"}
    x_pos = {"top": 0, "bottom": 1, "random": 2}

    for cond, grp in df.groupby("condition"):
        x = x_pos[cond]
        mean_kl = grp["kl"].mean()
        se_kl = grp["kl"].sem()
        ax.bar(x, mean_kl, color=colors[cond], alpha=0.8, label=cond_labels[cond], width=0.5)
        ax.errorbar(x, mean_kl, yerr=se_kl, color="black", capsize=5, linewidth=1.5)
        jitter = np.random.uniform(-0.12, 0.12, len(grp))
        ax.scatter(np.full(len(grp), x) + jitter, grp["kl"], color="black", s=15, alpha=0.4, zorder=5)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([cond_labels[c] for c in ("top", "bottom", "random")], fontsize=10)
    ax.set_ylabel("KL Divergence from Baseline", fontsize=11)
    ax.set_title(
        f"Output Change by Masking Condition  (n={df['sample_idx'].nunique()} samples, k={K_MASK})",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved KL comparison chart → %s", save_path)


def plot_flip_prob_drop(df: pd.DataFrame, save_path: Path) -> None:
    """Grouped bar chart: flip rate and probability drop by condition.

    Args:
        df: DataFrame with ``condition``, ``flip``, ``prob_drop`` columns.
        save_path: Output PNG path.
    """
    conds = ["top", "bottom", "random"]
    cond_labels = ["Top-k (条件A)", "Bottom-k (条件B)", "Random-k (条件C)"]
    flip_means = [df[df.condition == c]["flip"].mean() for c in conds]
    pdrop_means = [df[df.condition == c]["prob_drop"].mean() for c in conds]

    x = np.arange(len(conds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, flip_means, width=w, color="steelblue", alpha=0.85, label="Decision Flip Rate")
    ax.bar(x + w / 2, pdrop_means, width=w, color="salmon", alpha=0.85, label="Probability Drop")

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylabel("Rate / Drop", fontsize=11)
    ax.set_title(
        f"Flip Rate & Probability Drop by Condition  (n={df['sample_idx'].nunique()} samples)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved flip/prob-drop chart → %s", save_path)


def plot_group_scores_heatmap(
    group_matrix: np.ndarray,
    save_path: Path,
    n_samples: int,
) -> None:
    """Seaborn heatmap of per-sample baseline group importance.

    Args:
        group_matrix: ``(n_samples, 8)`` float array of normalized scores.
        save_path: Output PNG path.
        n_samples: Used for y-axis tick labels.
    """
    fig_h = max(4, n_samples * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    sns.heatmap(
        group_matrix,
        ax=ax,
        xticklabels=FEATURE_GROUP_NAMES,
        yticklabels=[f"Game {i}" for i in range(n_samples)],
        cmap="YlOrRd",
        annot=(n_samples <= 30),
        fmt=".2f",
        linewidths=0.3,
        linecolor="lightgrey",
        cbar_kws={"label": "Normalized Attention Score"},
    )
    ax.set_title(
        "Baseline Attention Group Importance  (last layer, mean over heads & queries)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved group heatmap → %s", save_path)


def plot_group_kl_breakdown(df: pd.DataFrame, save_path: Path) -> None:
    """Stacked bar showing which feature groups were most often masked in top-k condition,
    annotated with per-group mean KL contribution estimate.

    Args:
        df: Full results DataFrame.
        save_path: Output PNG path.
    """
    top_df = df[df.condition == "top"].copy()
    group_cols = [c for c in top_df.columns if c.startswith("masked_group_")]

    if not group_cols:
        logger.warning("No masked_group_* columns found, skipping group breakdown plot.")
        return

    # Count how often each group appears in the top-k mask
    counts = {g: 0 for g in FEATURE_GROUP_NAMES}
    for _, row in top_df.iterrows():
        for col in group_cols:
            g = row[col]
            if isinstance(g, str) and g in counts:
                counts[g] += 1

    names = list(counts)
    vals = [counts[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(names, vals, color="tomato", alpha=0.8)
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlabel("Times selected in Top-k mask", fontsize=10)
    ax.set_title(
        f"Feature groups selected as Top-{K_MASK} most important (条件A)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved group breakdown chart → %s", save_path)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full attention-group masking experiment."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ model
    model = load_model(CHECKPOINT, device)

    # ---------------------------------------------------------- parse XML rows
    logger.info("Parsing XML files from %s ...", XML_DIR)
    rows, report = build_dataset_rows_from_xml(XML_DIR, limit_files=N_XML_FILES)
    logger.info(
        "Extraction: files=%d  rounds=%d  samples=%d  errors=%d",
        report.files_processed,
        report.rounds_processed,
        len(rows),
        report.skipped_parse_errors,
    )
    if not rows:
        logger.error("No rows extracted — check XML path and format.")
        return

    # Limit to MAX_SAMPLES (sample uniformly if more available)
    if len(rows) > MAX_SAMPLES:
        rows = random.sample(rows, MAX_SAMPLES)
    logger.info("Running experiment on %d samples ...", len(rows))

    # ----------------------------------------------------------- run experiment
    all_results: List[Dict[str, Any]] = []
    group_matrix: List[List[float]] = []

    for idx, row in enumerate(rows):
        gs = row_to_game_state(row, device)
        try:
            result = evaluate_sample(model, gs, k=K_MASK, device=device)
        except Exception:  # noqa: BLE001
            logger.exception("Skipping sample %d", idx)
            continue

        group_matrix.append([result["group_scores"][g] for g in FEATURE_GROUP_NAMES])

        for cond, cres in result["condition_results"].items():
            row_out: Dict[str, Any] = {
                "sample_idx": idx,
                "condition": cond,
                "player_id": result["player_id"],
                "label": result["label"],
                "kl": cres["kl"],
                "flip": cres["flip"],
                "prob_drop": cres["prob_drop"],
                "n_positions_masked": cres["n_positions_masked"],
            }
            for g in FEATURE_GROUP_NAMES:
                row_out[f"baseline_{g}"] = result["group_scores"][g]
            for i, g in enumerate(cres["masked_groups"]):
                row_out[f"masked_group_{i}"] = g
            all_results.append(row_out)

        if (idx + 1) % 10 == 0:
            logger.info("  %d / %d done", idx + 1, len(rows))

    if not all_results:
        logger.error("No results — cannot plot or save.")
        return

    df = pd.DataFrame(all_results)
    n_valid = df["sample_idx"].nunique()
    logger.info("Completed %d samples.", n_valid)

    # -------------------------------------------------------- print summary
    print("\n" + "=" * 60)
    print(f"Attention Group Masking Experiment  (k={K_MASK}, n={n_valid})")
    print("=" * 60)
    for cond in ("top", "bottom", "random"):
        sub = df[df.condition == cond]
        print(
            f"  {cond:8s}  "
            f"KL={sub['kl'].mean():.4f}±{sub['kl'].std():.4f}  "
            f"FlipRate={sub['flip'].mean():.3f}  "
            f"ProbDrop={sub['prob_drop'].mean():.4f}"
        )

    print("\nTop-3 most important groups (mean baseline score):")
    baseline_cols = [c for c in df.columns if c.startswith("baseline_")]
    mean_scores = {c.replace("baseline_", ""): df[c].mean() for c in baseline_cols}
    for g, s in sorted(mean_scores.items(), key=lambda x: -x[1])[:3]:
        print(f"  {g:<22}: {s:.4f}")

    # ---------------------------------------------------------------- save CSV
    csv_path = RESULTS_DIR / "attn_group_mask_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Saved CSV → %s", csv_path)

    # --------------------------------------------------------------- save JSON
    summary = {}
    for cond in ("top", "bottom", "random"):
        sub = df[df.condition == cond]
        summary[cond] = {
            "mean_kl": round(float(sub["kl"].mean()), 5),
            "std_kl": round(float(sub["kl"].std()), 5),
            "mean_flip_rate": round(float(sub["flip"].mean()), 4),
            "mean_prob_drop": round(float(sub["prob_drop"].mean()), 5),
        }
    summary["experiment_config"] = {
        "n_samples": n_valid,
        "k_mask": K_MASK,
        "n_xml_files": N_XML_FILES,
        "checkpoint": CHECKPOINT.name,
        "model_config": {
            "d_model": model.config.d_model,
            "n_heads": model.config.n_heads,
            "n_layers": model.config.n_layers,
        },
        "random_seed": RANDOM_SEED,
    }
    json_path = RESULTS_DIR / "attn_group_mask_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved summary JSON → %s", json_path)

    # ------------------------------------------------------------ save figures
    plot_kl_comparison(df, FIGURES_DIR / "attn_mask_kl_comparison.png")
    plot_flip_prob_drop(df, FIGURES_DIR / "attn_mask_fliprate.png")
    plot_group_kl_breakdown(df, FIGURES_DIR / "attn_mask_group_breakdown.png")

    # Group heatmap via AttentionVisualizer
    gm = np.array(group_matrix)
    plot_group_scores_heatmap(gm, FIGURES_DIR / "attn_group_heatmap.png", n_samples=len(group_matrix))

    print(f"\nFigures saved to {FIGURES_DIR}/")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
