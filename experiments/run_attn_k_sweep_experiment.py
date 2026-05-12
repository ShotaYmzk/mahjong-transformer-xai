#!/usr/bin/env python3
"""k-sweep experiment: masking k=1..5 feature groups to justify k=3.

Runs the attention-group masking experiment for k ∈ {1, 2, 3, 4, 5} and
produces a comprehensive statistical analysis to justify the choice of k.

Key analyses
------------
1. **AOPC curve** — mean KL divergence vs k for top/bottom/random conditions.
   Faithful attention should show a monotonically increasing curve for top-k.
2. **Faithfulness gap** — KL(top-k) − KL(random-k).  Captures how much more
   damage "informed" masking does vs chance.  A peak here identifies the most
   discriminating k.
3. **Paired t-test** (``scipy.stats.ttest_rel``) — tests whether KL(top-k) >
   KL(random-k) at each k.  The smallest k with p < 0.05 shows when the
   effect becomes statistically detectable.
4. **Flip-rate & probability-drop curves** — corroborating evidence.
5. **Signal-to-noise ratio** — gap / (std of random-k KL).

All results are saved to ``outputs/`` and documented in
``docs/experiment-attn-group-mask-2026-05-12.md``.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data.observation_schema import build_dataset_rows_from_xml, DatasetRow  # noqa: E402
from models.mahjong_transformer_v2 import MahjongTransformerConfig, MahjongTransformerV2  # noqa: E402
from experiments.metrics.faithfulness import (  # noqa: E402
    decision_flip_rate,
    kl_divergence,
    probability_drop,
)
from attention_patching import FEATURE_GROUP_NAMES, _default_feature_group_map  # noqa: E402

# Import shared helpers from the initial experiment
from experiments.run_attn_group_mask_experiment import (  # noqa: E402
    load_model,
    row_to_game_state,
    build_position_patch,
    select_groups,
)

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
FIGURES_DIR = _ROOT / "outputs/figures"
RESULTS_DIR = _ROOT / "outputs/results"

N_XML_FILES = 5
MAX_SAMPLES = 50
K_VALUES = [1, 2, 3, 4, 5]
RANDOM_SEED = 42
N_RANDOM_REPEATS = 5   # repeat random condition to reduce variance

# Report / slide narrative: k maximizing Gap or SNR is often 2; we still adopt k=3
# (paired t power, interpretability, covers top-3 feature groups).  Plots mark both.
ADOPTED_K_FOR_ANALYSIS = 3


# ---------------------------------------------------------------------------
# Per-sample evaluation for a specific k
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sample_k(
    model: MahjongTransformerV2,
    game_state: Dict[str, Any],
    k: int,
    device: torch.device,
    random_repeats: int = N_RANDOM_REPEATS,
) -> Dict[str, Any]:
    """Run baseline + three masking conditions for one sample at a given k.

    The random condition is repeated ``random_repeats`` times and the results
    are averaged to reduce variance from the random draw.

    Args:
        model: Loaded model in eval mode.
        game_state: Dict from ``row_to_game_state``.
        k: Number of groups to mask.
        device: Inference device.
        random_repeats: How many times to repeat the random condition.

    Returns:
        Dict with per-condition ``kl``, ``flip``, ``prob_drop`` and
        ``group_scores`` (baseline importance).
    """
    cfg = model.config
    seq = game_state["sequence"]
    pid = game_state["player_id"]

    # Baseline forward
    logits_base, internals = model(
        game_state["static"], seq,
        game_state["hand_counts"], game_state["aka_flags"], game_state["valid_mask"],
        return_internals=True,
    )

    # Group scores from last layer
    last_attn = internals["attn_weights"][-1]
    importance = last_attn.mean(dim=(1, 2)).squeeze(0)
    group_positions = _default_feature_group_map(seq[0].cpu(), pid)

    raw_scores: Dict[str, float] = {}
    for name in FEATURE_GROUP_NAMES:
        pos = group_positions.get(name, [])
        raw_scores[name] = float(importance[pos].sum().item()) if pos else 0.0
    total = sum(raw_scores.values())
    group_scores = {n: v / total for n, v in raw_scores.items()} if total > 1e-8 else raw_scores

    condition_results: Dict[str, Dict[str, float]] = {}

    for cond in ("top", "bottom"):
        groups_to_mask = select_groups(group_scores, cond, k=k)
        positions_to_mask = sorted({p for g in groups_to_mask for p in group_positions.get(g, [])})
        patch = build_position_patch(positions_to_mask, 1, cfg.n_heads, seq.shape[1], device)
        logits_m = model(
            game_state["static"], seq,
            game_state["hand_counts"], game_state["aka_flags"], game_state["valid_mask"],
            attention_patch=patch,
        ) if patch else logits_base.clone()

        condition_results[cond] = {
            "kl": float(kl_divergence(logits_base, logits_m).item()),
            "flip": float(decision_flip_rate(logits_base, logits_m).item()),
            "prob_drop": float(probability_drop(logits_base, logits_m).item()),
            "n_positions": len(positions_to_mask),
        }

    # Average over multiple random draws
    rand_kls, rand_flips, rand_pdrops = [], [], []
    for _ in range(random_repeats):
        groups_r = select_groups(group_scores, "random", k=k)
        positions_r = sorted({p for g in groups_r for p in group_positions.get(g, [])})
        patch_r = build_position_patch(positions_r, 1, cfg.n_heads, seq.shape[1], device)
        logits_r = model(
            game_state["static"], seq,
            game_state["hand_counts"], game_state["aka_flags"], game_state["valid_mask"],
            attention_patch=patch_r,
        ) if patch_r else logits_base.clone()
        rand_kls.append(float(kl_divergence(logits_base, logits_r).item()))
        rand_flips.append(float(decision_flip_rate(logits_base, logits_r).item()))
        rand_pdrops.append(float(probability_drop(logits_base, logits_r).item()))

    condition_results["random"] = {
        "kl": float(np.mean(rand_kls)),
        "flip": float(np.mean(rand_flips)),
        "prob_drop": float(np.mean(rand_pdrops)),
        "n_positions": 0,
    }

    return {"group_scores": group_scores, "condition_results": condition_results}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _mean_std(df: pd.DataFrame, cond: str, metric: str, k: int) -> Tuple[float, float]:
    sub = df[(df.condition == cond) & (df.k == k)][metric]
    return float(sub.mean()), float(sub.sem())


def plot_aopc_curve(df: pd.DataFrame, save_path: Path) -> None:
    """AOPC-style line plot: mean KL vs k for all three conditions.

    Args:
        df: Full results DataFrame with ``k``, ``condition``, ``kl`` columns.
        save_path: Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "top":    ("tomato",         "o-",  "Top-k masked (条件A)"),
        "bottom": ("mediumseagreen", "s--", "Bottom-k masked (条件B)"),
        "random": ("darkorange",     "^:",  "Random-k masked (条件C)"),
    }
    for cond, (color, ls, label) in styles.items():
        means, errs = [], []
        for k in K_VALUES:
            m, e = _mean_std(df, cond, "kl", k)
            means.append(m)
            errs.append(e)
        ax.plot(K_VALUES, means, ls, color=color, label=label, linewidth=2, markersize=7)
        ax.fill_between(
            K_VALUES,
            [m - e for m, e in zip(means, errs)],
            [m + e for m, e in zip(means, errs)],
            color=color, alpha=0.15,
        )

    ax.set_xlabel("k  (number of feature groups masked)", fontsize=11)
    ax.set_ylabel("Mean KL Divergence from Baseline", fontsize=11)
    ax.set_title(
        "AOPC-style Curve: Output Change vs Masking Intensity\n"
        f"(n={df['sample_idx'].nunique()} samples, {N_RANDOM_REPEATS}× random avg)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved AOPC curve → %s", save_path)


def plot_faithfulness_gap(
    df: pd.DataFrame,
    save_path: Path,
    *,
    adopted_k: int = ADOPTED_K_FOR_ANALYSIS,
) -> int:
    """Bar chart of faithfulness gap = KL(top-k) − KL(random-k) per k.

    **Gold** fill marks the k with the largest mean gap (often k=2 here).  A **green
    outline** marks ``adopted_k`` (k=3 in our slides: paired t / interpretability /
    covering top-3 feature groups).  Those criteria need not pick the same k.

    Args:
        df: Full results DataFrame.
        save_path: Output PNG path.
        adopted_k: k highlighted as adopted for reporting (not necessarily argmax gap).

    Returns:
        k that maximizes the mean gap.
    """
    gaps, gap_errs = [], []
    for k in K_VALUES:
        paired = pd.merge(
            df[(df.condition == "top") & (df.k == k)][["sample_idx", "kl"]].rename(columns={"kl": "top"}),
            df[(df.condition == "random") & (df.k == k)][["sample_idx", "kl"]].rename(columns={"kl": "rnd"}),
            on="sample_idx",
        )
        diffs = paired["top"] - paired["rnd"]
        gaps.append(float(diffs.mean()))
        gap_errs.append(float(diffs.sem()))

    best_k = K_VALUES[int(np.argmax(gaps))]
    colors = ["gold" if k == best_k else "steelblue" for k in K_VALUES]
    ecols: List[str] = []
    ew: List[float] = []
    for k in K_VALUES:
        if k == adopted_k:
            ecols.append("darkgreen")
            ew.append(2.8)
        else:
            ecols.append("black")
            ew.append(0.7)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(
        K_VALUES,
        gaps,
        yerr=gap_errs,
        capsize=5,
        color=colors,
        alpha=0.85,
        edgecolor=ecols,
        linewidth=ew,
    )
    ax.bar_label(bars, labels=[f"{g:.4f}" for g in gaps], padding=4, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("k  (number of feature groups masked)", fontsize=11)
    ax.set_ylabel("KL(top-k) − KL(random-k)", fontsize=11)
    ax.set_title("Faithfulness Gap: Informed vs Random Masking", fontsize=11, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        f"Gold = largest mean gap (k={best_k}).  Green outline = adopted k={adopted_k} "
        "(main report: higher paired-t / covers 3 interpretable groups).",
        ha="center",
        fontsize=8.5,
    )
    ax.set_xticks(K_VALUES)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    leg_el = [
        Patch(facecolor="gold", edgecolor="black", linewidth=0.7, label=f"Largest mean gap (k={best_k})"),
        Patch(facecolor="steelblue", edgecolor="darkgreen", linewidth=2.5, label=f"Adopted k={adopted_k}"),
    ]
    ax.legend(handles=leg_el, loc="upper right", fontsize=8)
    fig.subplots_adjust(bottom=0.19)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved faithfulness gap chart → %s", save_path)
    return best_k


def plot_flip_and_pdrop_curves(df: pd.DataFrame, save_path: Path) -> None:
    """Two-panel line plot: flip rate and probability drop vs k.

    Args:
        df: Full results DataFrame.
        save_path: Output PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    styles = {
        "top":    ("tomato",         "o-",  "Top-k (条件A)"),
        "bottom": ("mediumseagreen", "s--", "Bottom-k (条件B)"),
        "random": ("darkorange",     "^:",  "Random-k (条件C)"),
    }
    for ax, metric, title, ylabel in [
        (axes[0], "flip",      "Decision Flip Rate vs k",      "Flip Rate"),
        (axes[1], "prob_drop", "Probability Drop vs k",        "Probability Drop"),
    ]:
        for cond, (color, ls, label) in styles.items():
            means = [_mean_std(df, cond, metric, k)[0] for k in K_VALUES]
            errs  = [_mean_std(df, cond, metric, k)[1] for k in K_VALUES]
            ax.plot(K_VALUES, means, ls, color=color, label=label, linewidth=2, markersize=6)
            ax.fill_between(K_VALUES,
                            [m - e for m, e in zip(means, errs)],
                            [m + e for m, e in zip(means, errs)],
                            color=color, alpha=0.12)
        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(K_VALUES)
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved flip/pdrop curves → %s", save_path)


def plot_snr(
    df: pd.DataFrame,
    save_path: Path,
    *,
    adopted_k: int = ADOPTED_K_FOR_ANALYSIS,
) -> None:
    """Signal-to-noise ratio: gap / std(random-KL) per k.

    Gold fill = k with largest SNR (usually k=2 here).  Green outline = ``adopted_k``
    for the written report (same convention as ``plot_faithfulness_gap``).
    """
    snrs = []
    for k in K_VALUES:
        top_kl  = df[(df.condition == "top")    & (df.k == k)]["kl"].values
        rnd_kl  = df[(df.condition == "random") & (df.k == k)]["kl"].values
        gap     = top_kl.mean() - rnd_kl.mean()
        noise   = rnd_kl.std() + 1e-9
        snrs.append(gap / noise)

    best_k = K_VALUES[int(np.argmax(snrs))]
    colors = ["gold" if k == best_k else "mediumpurple" for k in K_VALUES]
    ecols = ["darkgreen" if k == adopted_k else "black" for k in K_VALUES]
    ew = [2.8 if k == adopted_k else 0.7 for k in K_VALUES]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(K_VALUES, snrs, color=colors, alpha=0.85, edgecolor=ecols, linewidth=ew)
    ax.bar_label(bars, labels=[f"{s:.2f}" for s in snrs], padding=4, fontsize=9)
    ax.set_xlabel("k  (number of feature groups masked)", fontsize=11)
    ax.set_ylabel("SNR  =  gap / std(random KL)", fontsize=11)
    ax.set_title("Signal-to-Noise Ratio of Faithfulness Gap", fontsize=11, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        f"Gold = largest SNR (k={best_k}).  Green outline = adopted k={adopted_k}.",
        ha="center",
        fontsize=8.5,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(K_VALUES)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    leg_el = [
        Patch(facecolor="gold", edgecolor="black", linewidth=0.7, label=f"Largest SNR (k={best_k})"),
        Patch(facecolor="mediumpurple", edgecolor="darkgreen", linewidth=2.5, label=f"Adopted k={adopted_k}"),
    ]
    ax.legend(handles=leg_el, loc="upper right", fontsize=8)
    fig.subplots_adjust(bottom=0.17)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved SNR chart → %s", save_path)


def plot_pvalue_heatmap(pvalue_table: Dict[int, Dict[str, float]], save_path: Path) -> None:
    """Heatmap of paired t-test p-values (top vs random) for each k.

    Args:
        pvalue_table: ``{k: {"t": ..., "p": ..., "significant": ...}}``.
        save_path: Output PNG path.
    """
    ks = sorted(pvalue_table)
    pvals = [pvalue_table[k]["p"] for k in ks]
    tvals = [pvalue_table[k]["t"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # p-value bar
    ax = axes[0]
    bar_colors = ["tomato" if p < 0.05 else "lightgrey" for p in pvals]
    bars = ax.bar(ks, pvals, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, labels=[f"{p:.3f}" for p in pvals], padding=3, fontsize=9)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.2, label="p=0.05 threshold")
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("p-value  (one-tailed t-test: top > random)", fontsize=10)
    ax.set_title("Statistical Significance of Top-k vs Random-k", fontsize=11, fontweight="bold")
    ax.set_xticks(ks)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # t-statistic bar
    ax2 = axes[1]
    bars2 = ax2.bar(ks, tvals, color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.7)
    ax2.bar_label(bars2, labels=[f"{t:.2f}" for t in tvals], padding=3, fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("k", fontsize=11)
    ax2.set_ylabel("t-statistic", fontsize=11)
    ax2.set_title("t-statistic: top-k KL vs random-k KL", fontsize=11, fontweight="bold")
    ax2.set_xticks(ks)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved p-value chart → %s", save_path)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def run_statistical_tests(df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """Run paired t-tests (top-k vs random-k) for each k value.

    Args:
        df: Full results DataFrame.

    Returns:
        Dict ``{k: {"t": t_stat, "p": one_tailed_p, "significant": bool}}``.
    """
    results = {}
    for k in K_VALUES:
        merged = pd.merge(
            df[(df.condition == "top")    & (df.k == k)][["sample_idx", "kl"]].rename(columns={"kl": "top"}),
            df[(df.condition == "random") & (df.k == k)][["sample_idx", "kl"]].rename(columns={"kl": "rnd"}),
            on="sample_idx",
        )
        if len(merged) < 2:
            results[k] = {"t": 0.0, "p": 1.0, "significant": False}
            continue
        t_stat, p_two = stats.ttest_rel(merged["top"], merged["rnd"])
        # One-tailed: H1 = top > random
        p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
        results[k] = {
            "t": round(float(t_stat), 4),
            "p": round(float(p_one), 4),
            "significant": p_one < 0.05,
        }
    return results


def build_summary_table(df: pd.DataFrame, pvalue_table: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    """Build a comprehensive summary table across k values and conditions.

    Args:
        df: Full results DataFrame.
        pvalue_table: Output of :func:`run_statistical_tests`.

    Returns:
        DataFrame with one row per k value, columns for all conditions.
    """
    rows = []
    for k in K_VALUES:
        row: Dict[str, Any] = {"k": k}
        for cond in ("top", "bottom", "random"):
            sub = df[(df.condition == cond) & (df.k == k)]
            row[f"{cond}_kl_mean"]    = round(float(sub["kl"].mean()), 5)
            row[f"{cond}_kl_std"]     = round(float(sub["kl"].std()), 5)
            row[f"{cond}_flip_mean"]  = round(float(sub["flip"].mean()), 4)
            row[f"{cond}_pdrop_mean"] = round(float(sub["prob_drop"].mean()), 5)

        # Gap = top_kl - random_kl
        paired = pd.merge(
            df[(df.condition == "top")    & (df.k == k)][["sample_idx", "kl"]].rename(columns={"kl": "top"}),
            df[(df.condition == "random") & (df.k == k)][["sample_idx", "kl"]].rename(columns={"kl": "rnd"}),
            on="sample_idx",
        )
        diffs = paired["top"] - paired["rnd"]
        row["faithfulness_gap"]    = round(float(diffs.mean()), 5)
        row["snr"]                 = round(float(diffs.mean() / (df[(df.condition == "random") & (df.k == k)]["kl"].std() + 1e-9)), 3)
        row["t_stat"]              = pvalue_table[k]["t"]
        row["p_value_onetail"]     = pvalue_table[k]["p"]
        row["significant_p05"]     = pvalue_table[k]["significant"]
        rows.append(row)

    return pd.DataFrame(rows)


def regenerate_figures_from_disk() -> None:
    """Redraw k-sweep figures from ``outputs/results/attn_k_sweep_results.csv`` (no model run)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "attn_k_sweep_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))
    df = pd.read_csv(csv_path)
    pvalue_table = run_statistical_tests(df)
    plot_aopc_curve(df, FIGURES_DIR / "attn_k_sweep_aopc_curve.png")
    plot_faithfulness_gap(df, FIGURES_DIR / "attn_k_sweep_faithfulness_gap.png")
    plot_flip_and_pdrop_curves(df, FIGURES_DIR / "attn_k_sweep_flip_pdrop.png")
    plot_snr(df, FIGURES_DIR / "attn_k_sweep_snr.png")
    plot_pvalue_heatmap(pvalue_table, FIGURES_DIR / "attn_k_sweep_pvalues.png")
    print(f"Figures refreshed → {FIGURES_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the k-sweep experiment."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(CHECKPOINT, device)

    # Parse XML
    logger.info("Parsing XML ...")
    rows, report = build_dataset_rows_from_xml(XML_DIR, limit_files=N_XML_FILES)
    logger.info("Extracted %d samples from %d files.", len(rows), report.files_processed)
    if not rows:
        logger.error("No rows found.")
        return
    if len(rows) > MAX_SAMPLES:
        rows = random.sample(rows, MAX_SAMPLES)
    n_samples = len(rows)
    logger.info("Using %d samples, k ∈ %s", n_samples, K_VALUES)

    # Pre-compute game states
    game_states = [row_to_game_state(r, device) for r in rows]

    # Sweep
    all_rows: List[Dict[str, Any]] = []
    for k in K_VALUES:
        logger.info("k=%d ...", k)
        for idx, gs in enumerate(game_states):
            try:
                res = evaluate_sample_k(model, gs, k=k, device=device)
            except Exception:  # noqa: BLE001
                logger.exception("Sample %d k=%d failed, skipping.", idx, k)
                continue
            for cond, cres in res["condition_results"].items():
                r: Dict[str, Any] = {
                    "sample_idx": idx, "k": k, "condition": cond,
                    "kl": cres["kl"], "flip": cres["flip"], "prob_drop": cres["prob_drop"],
                }
                for g in FEATURE_GROUP_NAMES:
                    r[f"score_{g}"] = res["group_scores"][g]
                all_rows.append(r)

    df = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / "attn_k_sweep_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Saved CSV → %s", csv_path)

    # Statistical tests
    pvalue_table = run_statistical_tests(df)

    # Summary table
    summary_df = build_summary_table(df, pvalue_table)
    summary_csv = RESULTS_DIR / "attn_k_sweep_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # Print table
    print("\n" + "=" * 78)
    print(f"k-sweep Experiment  (n={n_samples} samples  random_repeats={N_RANDOM_REPEATS})")
    print("=" * 78)
    header = f"{'k':>3}  {'KL_top':>8}  {'KL_bot':>8}  {'KL_rnd':>8}  {'Gap':>8}  {'SNR':>6}  {'t':>7}  {'p(1t)':>7}  {'sig':>4}"
    print(header)
    print("-" * 78)
    for _, row in summary_df.iterrows():
        sig = "***" if row.p_value_onetail < 0.01 else ("*" if row.p_value_onetail < 0.05 else "")
        print(
            f"{int(row.k):>3}  "
            f"{row.top_kl_mean:>8.4f}  "
            f"{row.bottom_kl_mean:>8.4f}  "
            f"{row.random_kl_mean:>8.4f}  "
            f"{row.faithfulness_gap:>8.4f}  "
            f"{row.snr:>6.2f}  "
            f"{row.t_stat:>7.3f}  "
            f"{row.p_value_onetail:>7.4f}  "
            f"{sig:>4}"
        )

    best_gap_k  = int(summary_df.loc[summary_df.faithfulness_gap.idxmax(), "k"])
    best_snr_k  = int(summary_df.loc[summary_df.snr.idxmax(), "k"])
    first_sig_k = next((int(r.k) for _, r in summary_df.iterrows() if r.significant_p05), None)

    print(f"\nBest faithfulness gap:     k = {best_gap_k}")
    print(f"Best SNR:                  k = {best_snr_k}")
    print(f"First significant (p<.05): k = {first_sig_k}")

    # Save summary JSON
    json_out = {
        "experiment_config": {
            "n_samples": n_samples, "k_values": K_VALUES,
            "n_xml_files": N_XML_FILES, "random_repeats": N_RANDOM_REPEATS,
            "checkpoint": CHECKPOINT.name, "random_seed": RANDOM_SEED,
            "model_config": {
                "d_model": model.config.d_model,
                "n_heads": model.config.n_heads,
                "n_layers": model.config.n_layers,
            },
        },
        "k_results": {
            int(row.k): {
                "top_kl_mean": row.top_kl_mean, "top_kl_std": row.top_kl_std,
                "bottom_kl_mean": row.bottom_kl_mean,
                "random_kl_mean": row.random_kl_mean, "random_kl_std": row.random_kl_std,
                "faithfulness_gap": row.faithfulness_gap,
                "snr": row.snr,
                "top_flip_mean": row.top_flip_mean,
                "t_stat": row.t_stat,
                "p_value_onetail": row.p_value_onetail,
                "significant": row.significant_p05,
            }
            for _, row in summary_df.iterrows()
        },
        "conclusions": {
            "best_gap_k": best_gap_k,
            "best_snr_k": best_snr_k,
            "first_significant_k": first_sig_k,
        },
    }
    json_path = RESULTS_DIR / "attn_k_sweep_summary.json"
    json_path.write_text(json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved summary JSON → %s", json_path)

    # Plots
    plot_aopc_curve(df, FIGURES_DIR / "attn_k_sweep_aopc_curve.png")
    plot_faithfulness_gap(df, FIGURES_DIR / "attn_k_sweep_faithfulness_gap.png")
    plot_flip_and_pdrop_curves(df, FIGURES_DIR / "attn_k_sweep_flip_pdrop.png")
    plot_snr(df, FIGURES_DIR / "attn_k_sweep_snr.png")
    plot_pvalue_heatmap(pvalue_table, FIGURES_DIR / "attn_k_sweep_pvalues.png")

    print(f"\nFigures → {FIGURES_DIR}/")
    print(f"Results → {RESULTS_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plots-only":
        regenerate_figures_from_disk()
    else:
        main()
