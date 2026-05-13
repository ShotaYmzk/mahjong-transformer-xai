#!/usr/bin/env python3
"""Top-k positional attention masking (no feature-group aggregation).

Runs the same faithfulness metrics as the group-mask experiment (KL divergence,
decision flip rate, probability drop) but selects key positions directly from
last-layer attention importance: top-k, bottom-k, or random positions among
non-padding sequence tokens.

Outputs
---------
- Timestamped JSONL under ``outputs/results/attn_topk_position_mask_<ts>.jsonl``.
- Aggregated ``outputs/results/attn_topk_position_summary.json`` and CSV
  ``attn_topk_position_results.csv`` (unless ``--no-plots``).
- Figures under ``figure/``: ``attn_topk_position_kl_vs_k.png``, bar charts for k=3.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data.observation_schema import build_dataset_rows_from_xml  # noqa: E402
from experiments.metrics.faithfulness import (  # noqa: E402
    decision_flip_rate,
    kl_divergence,
    probability_drop,
)
from experiments.run_attn_group_mask_experiment import (  # noqa: E402
    build_position_patch,
    load_model,
    row_to_game_state,
)
from models.mahjong_transformer_v2 import MahjongTransformerV2  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

_PADDING_EVENT_CODE = 8  # observation_schema.EVENT_TYPES["PADDING"]
_K_VALUES_DEFAULT = [1, 3, 5]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--xml-dir",
        type=Path,
        default=Path("/home/ubuntu/Documents/tenhou_xml_2023"),
        help="Tenhou XML directory or file path forwarded to dataset builder.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_ROOT / "outputs/impl1/hdf5_10epoch.pt",
        help="MahjongTransformerV2 checkpoint.",
    )
    p.add_argument(
        "--n-xml-files",
        type=int,
        default=5,
        help="Number of XML files to parse from the corpus (ordering as in schema).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=150,
        help="Maximum evaluation samples drawn after extraction.",
    )
    p.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=_K_VALUES_DEFAULT,
        help="Candidate k values (positional mask count); each is run in an outer loop.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (XML sampling and random-mask positions).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="'cuda', 'cuda:0', or 'cpu'.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "JSONL output path (default: outputs/results/attn_topk_position_mask_<timestamp>.jsonl)."
        ),
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib summary figures and aggregated JSON.",
    )
    return p.parse_args()


def non_padding_positions(seq_row: torch.Tensor) -> List[int]:
    """Return sorted indices where the event type is not PADDING."""
    codes = seq_row[:, 0].round().long()
    positions = torch.nonzero(codes != _PADDING_EVENT_CODE, as_tuple=False).squeeze(-1)
    if positions.numel() == 0:
        return []
    return positions.cpu().tolist()


def rng_for_random_positions(base_seed: int, sample_idx: int, k: int) -> random.Random:
    """Deterministic RNG for random-k selection (no ``hash()`` — PEP 456 safe)."""
    mixed = (
        int(base_seed) ^ (sample_idx * 0x9E3779B9) ^ (k * 0x85EBCA6B) ^ 0xC2B2AE3D
    ) & 0xFFFFFFFF
    return random.Random(mixed)


@torch.no_grad()
def evaluate_sample(
    model: MahjongTransformerV2,
    game_state: Dict[str, Any],
    *,
    k: int,
    device: torch.device,
    seed: int,
    sample_idx: int,
) -> Dict[str, Any]:
    """Baseline forward + three position-based masks for fixed k."""
    cfg = model.config
    seq = game_state["sequence"]

    logits_base, internals = model(
        game_state["static"],
        seq,
        game_state["hand_counts"],
        game_state["aka_flags"],
        game_state["valid_mask"],
        return_internals=True,
    )

    last_attn = internals["attn_weights"][-1]
    importance = last_attn.mean(dim=(1, 2)).squeeze(0)

    valid = non_padding_positions(seq[0].cpu())
    take = min(int(k), len(valid)) if valid else 0

    def select_positions(condition: str) -> List[int]:
        if take == 0 or not valid:
            return []
        if condition == "top":
            vt = torch.tensor(valid, dtype=torch.long, device=importance.device)
            vals = importance[vt]
            _, sel = torch.topk(vals, take, largest=True)
            return sorted(int(vt[i].item()) for i in sel)
        if condition == "bottom":
            vt = torch.tensor(valid, dtype=torch.long, device=importance.device)
            vals = importance[vt]
            _, sel = torch.topk(vals, take, largest=False)
            return sorted(int(vt[i].item()) for i in sel)
        if condition == "random":
            rng = rng_for_random_positions(seed, sample_idx, k)
            return sorted(rng.sample(valid, take))
        raise ValueError(f"Unknown condition: {condition!r}")

    condition_results: Dict[str, Dict[str, Any]] = {}

    for cond in ("top", "bottom", "random"):
        positions_to_mask = select_positions(cond)
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

        imp_masked = [float(importance[pos].item()) for pos in positions_to_mask]

        kl = float(kl_divergence(logits_base, logits_masked).item())
        flip = float(decision_flip_rate(logits_base, logits_masked).item())
        pdrop = float(probability_drop(logits_base, logits_masked).item())

        condition_results[cond] = {
            "kl": kl,
            "flip": flip,
            "prob_drop": pdrop,
            "k": int(k),
            "k_effective": take,
            "masked_positions": positions_to_mask,
            "importance_scores_masked": imp_masked,
            "n_positions_masked": len(positions_to_mask),
        }

    return {
        "condition_results": condition_results,
        "label": game_state["label"],
        "player_id": game_state["player_id"],
        "non_padding_len": len(valid),
    }


FIGURES_DIR = _ROOT / "figure"
RESULTS_DIR = _ROOT / "outputs" / "results"


def default_jsonl_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"attn_topk_position_mask_{ts}.jsonl"


def build_summary_and_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate means and paired top vs random t-test per k."""
    summary: Dict[str, Any] = {"by_k_condition": {}, "paired_top_vs_random": {}}
    for k in sorted(df["k"].unique()):
        sub_k = "by_k_condition"
        summary[sub_k][str(int(k))] = {}
        for cond in ("top", "bottom", "random"):
            sub = df[(df["k"] == k) & (df["condition"] == cond)]
            summary[sub_k][str(int(k))][cond] = {
                "mean_kl": round(float(sub["kl"].mean()), 5),
                "std_kl": round(float(sub["kl"].std()), 5),
                "mean_flip": round(float(sub["flip"].mean()), 4),
                "mean_prob_drop": round(float(sub["prob_drop"].mean()), 5),
                "n": int(len(sub)),
            }
        merged = pd.merge(
            df[(df["k"] == k) & (df["condition"] == "top")][["sample_idx", "kl"]].rename(
                columns={"kl": "top_kl"}
            ),
            df[(df["k"] == k) & (df["condition"] == "random")][["sample_idx", "kl"]].rename(
                columns={"kl": "rnd_kl"}
            ),
            on="sample_idx",
        )
        if len(merged) >= 2:
            t_stat, p_two = stats.ttest_rel(merged["top_kl"], merged["rnd_kl"])
            p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
            gap = float(merged["top_kl"].mean() - merged["rnd_kl"].mean())
            summary["paired_top_vs_random"][str(int(k))] = {
                "faithfulness_gap": round(gap, 5),
                "t_stat": round(float(t_stat), 4),
                "p_value_onetail": round(float(p_one), 6),
            }
        else:
            summary["paired_top_vs_random"][str(int(k))] = {
                "faithfulness_gap": 0.0,
                "t_stat": 0.0,
                "p_value_onetail": 1.0,
            }
    return summary


def save_plots(df: pd.DataFrame, prefix: str = "attn_topk_position") -> None:
    """Write KL curve and k=3 condition bars (positional masking)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ks = sorted(df["k"].unique())
    conds = ["top", "bottom", "random"]
    colors = {"top": "tomato", "bottom": "mediumseagreen", "random": "darkorange"}

    # Fig 1: mean KL vs k per condition (line plot)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for c in conds:
        means = [df[(df["k"] == k) & (df["condition"] == c)]["kl"].mean() for k in ks]
        ax1.plot(ks, means, marker="o", linewidth=2, label=c, color=colors[c])
    ax1.set_xlabel("k  (masked key positions)")
    ax1.set_ylabel("Mean KL divergence")
    ax1.set_title(f"Top-k positional mask vs k  (n={df['sample_idx'].nunique()} samples)")
    ax1.legend()
    ax1.grid(alpha=0.35, linestyle="--")
    fig1.tight_layout()
    kl_curve = FIGURES_DIR / f"{prefix}_kl_vs_k.png"
    fig1.savefig(kl_curve, dpi=150)
    plt.close(fig1)

    # Fig 2 & 3: prefer k=3 when present (aligns with group-mask report)
    k_bar = 3 if 3 in ks else ks[len(ks) // 2]
    df_bar = df[df["k"] == k_bar]
    if df_bar.empty:
        logger.warning("No rows for k=%s; skipping bar charts.", k_bar)
        logger.info("Saved figure: %s", kl_curve.name)
        return

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(conds))
    mean_kls = [df_bar[df_bar.condition == c]["kl"].mean() for c in conds]
    se_kls = [df_bar[df_bar.condition == c]["kl"].sem() for c in conds]
    bars = ax2.bar(x_pos, mean_kls, yerr=se_kls, capsize=5, alpha=0.85, color=[colors[c] for c in conds])
    ax2.bar_label(bars, labels=[f"{m:.4f}" for m in mean_kls], padding=4, fontsize=9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{c}\npositional" for c in ("Top-k", "Bottom-k", "Random-k")])
    ax2.set_ylabel("Mean KL divergence")
    ax2.set_title(f"Positional mask  (k={k_bar}, n={df_bar['sample_idx'].nunique()} samples)")
    ax2.grid(axis="y", alpha=0.35, linestyle="--")
    fig2.tight_layout()
    kl_bar_path = FIGURES_DIR / f"{prefix}_kl_bar_k{k_bar}.png"
    fig2.savefig(kl_bar_path, dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    flips = [df_bar[df_bar.condition == c]["flip"].mean() for c in conds]
    pdrops = [df_bar[df_bar.condition == c]["prob_drop"].mean() for c in conds]
    w = 0.35
    ax3.bar(x_pos - w / 2, flips, width=w, label="Flip rate", color="steelblue", alpha=0.85)
    ax3.bar(x_pos + w / 2, pdrops, width=w, label="Prob drop", color="salmon", alpha=0.85)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{c}" for c in ("Top-k", "Bottom-k", "Random-k")])
    ax3.set_ylabel("Mean")
    ax3.set_title(f"Flip rate & Prob drop (positional, k={k_bar})")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.35, linestyle="--")
    fig3.tight_layout()
    flip_path = FIGURES_DIR / f"{prefix}_flip_pdrop_k{k_bar}.png"
    fig3.savefig(flip_path, dpi=150)
    plt.close(fig3)
    logger.info("Saved figures: %s, %s, %s", kl_curve.name, kl_bar_path.name, flip_path.name)


def main() -> None:
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device)
    logger.info("Device: %s", device)

    k_values = sorted({int(x) for x in args.k_values if int(x) > 0})
    if not k_values:
        raise SystemExit("--k-values must contain at least one positive integer.")

    out_path = args.output if args.output is not None else default_jsonl_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("JSONL → %s", out_path.resolve())

    model = load_model(args.checkpoint, device)

    logger.info("Parsing XML files from %s ...", args.xml_dir)
    rows, report = build_dataset_rows_from_xml(args.xml_dir, limit_files=args.n_xml_files)
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

    if len(rows) > args.max_samples:
        rows = random.sample(rows, args.max_samples)
    logger.info("Evaluating %d samples; k_values=%s …", len(rows), k_values)

    n_written = 0
    all_records: List[Dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            gs = row_to_game_state(row, device)
            for k in k_values:
                try:
                    result = evaluate_sample(
                        model, gs, k=k, device=device, seed=seed, sample_idx=idx
                    )
                except Exception:
                    logger.exception("Skip sample_idx=%d k=%d", idx, k)
                    continue

                for cond, cres in result["condition_results"].items():
                    record = {
                        "sample_idx": idx,
                        "condition": cond,
                        "k": cres["k"],
                        "k_effective": cres["k_effective"],
                        "masked_positions": cres["masked_positions"],
                        "importance_scores_masked": cres["importance_scores_masked"],
                        "non_padding_len": result["non_padding_len"],
                        "player_id": result["player_id"],
                        "label": result["label"],
                        "kl": cres["kl"],
                        "flip": cres["flip"],
                        "prob_drop": cres["prob_drop"],
                        "n_positions_masked": cres["n_positions_masked"],
                        "checkpoint": args.checkpoint.name,
                        "seed": seed,
                        "n_xml_files": args.n_xml_files,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    all_records.append(record)
                    n_written += 1

            if (idx + 1) % 10 == 0:
                logger.info("  %d / %d samples done (%d rows written)", idx + 1, len(rows), n_written)

    logger.info("Wrote %d JSONL records to %s", n_written, out_path.resolve())
    print(f"Wrote {n_written} rows → {out_path.resolve()}")

    if not args.no_plots and all_records:
        df = pd.DataFrame(all_records)
        agg = build_summary_and_stats(df)
        agg["experiment_config"] = {
            "jsonl": str(out_path.name),
            "n_samples": int(df["sample_idx"].nunique()),
            "k_values": k_values,
            "checkpoint": args.checkpoint.name,
            "seed": seed,
            "n_xml_files": args.n_xml_files,
            "device": str(device),
        }
        summary_path = RESULTS_DIR / "attn_topk_position_summary.json"
        summary_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Summary JSON → %s", summary_path)
        df.to_csv(RESULTS_DIR / "attn_topk_position_results.csv", index=False, encoding="utf-8-sig")
        logger.info("CSV → %s", RESULTS_DIR / "attn_topk_position_results.csv")
        save_plots(df)
        print(f"Summary JSON → {summary_path.resolve()}")


if __name__ == "__main__":
    main()
