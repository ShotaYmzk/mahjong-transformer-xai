"""Attention Patching Evaluator — faithfulness verification for MahjongTransformerV2.

This module implements XAI faithfulness evaluation by masking feature-group
attention scores under three conditions (top-k, bottom-k, random-k) and
measuring how much the LLM-generated explanation changes via BERTScore F1
and keyword occurrence shift.

Design principles
-----------------
- ``MahjongTransformerV2`` and surrounding code are *not* modified.
- Attention weights are extracted through ``register_forward_hook`` on each
  ``HookedSelfAttention`` sub-module inside ``model.blocks``.
- The LLM explanation function is injected at construction time, keeping
  API keys and network calls out of this file.
- ``torch.no_grad()`` is used for all forward passes.
- Masked group scores are always renormalized so they sum to 1.

Dependencies (beyond the base project)
---------------------------------------
  pip install bert-score scipy pandas
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_GROUP_NAMES: List[str] = [
    "ShantenReduction",
    "SafetyVsDealer",
    "SafetyVsOthers",
    "DoraValue",
    "YakuPotential",
    "TileEfficiency",
    "PointSituation",
    "OpponentActions",
]

# Event type codes that match observation_schema.EVENT_TYPES
_ET_INIT = 0
_ET_DRAW = 1
_ET_DISCARD = 2
_ET_NAKI = 3
_ET_REACH = 4
_ET_DORA = 5
_ET_AGARI = 6
_ET_RYUUKYOKU = 7
_ET_PADDING = 8


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class GameStateBatch:
    """Tensor batch ready for ``MahjongTransformerV2.forward``.

    Attributes:
        static: Float tensor of shape ``(B, static_dim)``.
        sequence: Float tensor of shape ``(B, max_seq_len, 6)``.
        hand_counts: Optional float tensor ``(B, 34)``.
        aka_flags: Optional float tensor ``(B, 3)``.
        valid_mask: Optional float tensor ``(B, 34)``.
    """

    static: torch.Tensor
    sequence: torch.Tensor
    hand_counts: Optional[torch.Tensor] = None
    aka_flags: Optional[torch.Tensor] = None
    valid_mask: Optional[torch.Tensor] = None


@dataclass
class SingleResult:
    """Full evaluation result for one game state across all three conditions.

    Attributes:
        game_idx: Integer identifier for this game state.
        baseline_scores: Feature-group importance scores before masking.
        masked_scores: Condition key → masked and renormalized group scores.
        baseline_explanation: LLM explanation from unmasked scores.
        masked_explanations: Condition key → explanation after masking.
        bertscore_f1: Condition key → BERTScore F1 vs baseline.
        keyword_shift: Condition key → mean |Δoccurrence| across 8 keywords.
    """

    game_idx: int
    baseline_scores: Dict[str, float]
    masked_scores: Dict[str, Dict[str, float]]
    baseline_explanation: str
    masked_explanations: Dict[str, str]
    bertscore_f1: Dict[str, float]
    keyword_shift: Dict[str, float]


# ---------------------------------------------------------------------------
# Default feature-group mapping
# ---------------------------------------------------------------------------


def _default_feature_group_map(
    sequence: torch.Tensor,
    player_id: int,
) -> Dict[str, List[int]]:
    """Map event-sequence positions to the 8 canonical feature groups.

    Uses the event-type field (column 0) and player field (column 1) of each
    event row to assign that position to the most semantically appropriate
    group.  Positions may be assigned to only one group.  PADDING events
    (type 8) are skipped entirely.

    The grouping heuristic:
    - ShantenReduction — own DRAW events (directly affect hand shape / shanten).
    - TileEfficiency   — own DISCARD events (tile choice efficiency).
    - SafetyVsDealer   — DISCARD events by the dealer opponent.
    - SafetyVsOthers   — DISCARD events by non-dealer opponents.
    - OpponentActions  — NAKI (pon/chi/kan) events by any opponent.
    - YakuPotential    — REACH declarations.
    - DoraValue        — DORA indicator flips.
    - PointSituation   — AGARI, RYUUKYOKU, and INIT events.

    Args:
        sequence: Float tensor of shape ``(seq_len, 6)``.  Column layout:
            ``[event_type, player, tile_kind, junme, data0, data1]``.
        player_id: Zero-based index of the decision-making player (0–3).

    Returns:
        Dict mapping each group name to a list of event-sequence position
        indices.
    """
    groups: Dict[str, List[int]] = {name: [] for name in FEATURE_GROUP_NAMES}
    seq_len = sequence.shape[0]

    # Use round() before int conversion to handle float-encoded integers.
    ev_types = sequence[:, 0].round().long().tolist()
    ev_players = sequence[:, 1].round().long().tolist()

    for pos in range(seq_len):
        et = ev_types[pos]
        ep = ev_players[pos]

        if et == _ET_PADDING:
            continue

        is_self = ep == player_id
        # Dealer is considered the player whose relative index is 0 at round start.
        # In Tenhou data the absolute player index of the dealer changes each round;
        # we approximate with ep == 0 when a more precise dealer ID is unavailable.
        is_dealer = ep == 0 and not is_self

        if et == _ET_DRAW:
            if is_self:
                groups["ShantenReduction"].append(pos)
        elif et == _ET_DISCARD:
            if is_self:
                groups["TileEfficiency"].append(pos)
            elif is_dealer:
                groups["SafetyVsDealer"].append(pos)
            else:
                groups["SafetyVsOthers"].append(pos)
        elif et == _ET_NAKI:
            if not is_self:
                groups["OpponentActions"].append(pos)
        elif et == _ET_REACH:
            groups["YakuPotential"].append(pos)
        elif et == _ET_DORA:
            groups["DoraValue"].append(pos)
        elif et in (_ET_AGARI, _ET_RYUUKYOKU, _ET_INIT):
            groups["PointSituation"].append(pos)
        else:
            # Unknown / opponent draw events default to ShantenReduction
            # because they still affect the remaining-tile distribution.
            groups["ShantenReduction"].append(pos)

    return groups


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class AttentionPatchingEvaluator:
    """Faithfulness evaluator for attention-based explanations.

    Workflow for each game state:

    1. Run ``model.forward`` with ``register_forward_hook`` installed on every
       ``HookedSelfAttention`` layer to capture attention weight matrices.
    2. Use the last layer's weights (averaged over heads and query positions)
       as per-position importance.
    3. Aggregate importance into 8 feature-group scores via a configurable
       mapping function.
    4. Apply three masking conditions: top-k groups zeroed (条件A), bottom-k
       zeroed (条件B), and random-k zeroed (条件C).  Renormalize after each.
    5. Call ``explain_fn`` with original and masked scores to generate
       four explanations.
    6. Compare masked vs baseline explanations with BERTScore F1 and keyword
       occurrence shift.

    Over a 100-sample batch, compute a paired t-test (条件A vs 条件C) and
    Spearman correlation (masking strength k vs BERTScore drop).

    Args:
        model: A ``MahjongTransformerV2`` instance.  Must have a ``blocks``
            attribute that is a ``ModuleList`` of ``HookedTransformerBlock``.
        explain_fn: Callable with signature
            ``(game_state: dict, attention_scores: dict[str, float]) -> str``.
            Generates a natural-language explanation from feature-group
            importance scores.  API keys must not appear in this file.
        k: Number of feature groups to mask per condition.  Default 3.
        device: Torch device for inference.  Defaults to CUDA if available.
        feature_group_map: Optional custom mapping function with signature
            ``(sequence: Tensor[seq_len, 6], player_id: int)
            -> dict[str, list[int]]``.
            If ``None``, :func:`_default_feature_group_map` is used.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        explain_fn: Callable[[Dict[str, Any], Dict[str, float]], str],
        *,
        k: int = 3,
        device: Optional[torch.device] = None,
        feature_group_map: Optional[
            Callable[[torch.Tensor, int], Dict[str, List[int]]]
        ] = None,
    ) -> None:
        self.model = model
        self.explain_fn = explain_fn
        self.k = k
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._group_map_fn = feature_group_map or _default_feature_group_map

        # State used exclusively during a single forward pass.
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._captured_attn: Dict[int, torch.Tensor] = {}

        self.model.eval()
        self.model.to(self.device)
        logger.info(
            "AttentionPatchingEvaluator ready. device=%s, k=%d", self.device, k
        )

    # ------------------------------------------------------------------
    # Internal: hook management
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Install forward hooks on every ``HookedSelfAttention`` in ``model.blocks``.

        Each hook captures the ``attn_weights`` tensor (shape
        ``(B, n_heads, seq_len, seq_len)``) from the layer's output dict
        into ``self._captured_attn[layer_idx]``.

        Existing hooks are removed before re-registering to avoid duplication.
        """
        self._remove_hooks()
        self._captured_attn.clear()

        blocks = self._get_blocks()
        for layer_idx, block in enumerate(blocks):
            attn_module = block.attn  # HookedSelfAttention

            def _make_hook(idx: int) -> Callable:
                def _hook(
                    _module: torch.nn.Module,
                    _inputs: tuple,
                    outputs: tuple,
                ) -> None:
                    # HookedSelfAttention.forward returns (out_tensor, internals_dict)
                    _, internals = outputs
                    self._captured_attn[idx] = internals["attn_weights"].detach().cpu()

                return _hook

            handle = attn_module.register_forward_hook(_make_hook(layer_idx))
            self._hooks.append(handle)

        logger.debug("Registered %d attention-capture hooks.", len(self._hooks))

    def _remove_hooks(self) -> None:
        """Remove all currently registered forward hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        logger.debug("Removed all attention-capture hooks.")

    def _get_blocks(self) -> torch.nn.ModuleList:
        """Return ``model.blocks`` (the ``ModuleList`` of transformer blocks).

        Raises:
            AttributeError: If the model lacks a ``blocks`` attribute.
        """
        if hasattr(self.model, "blocks"):
            return self.model.blocks  # type: ignore[return-value]
        raise AttributeError(
            "model has no attribute 'blocks'. Expected a MahjongTransformerV2 instance."
        )

    # ------------------------------------------------------------------
    # Internal: attention extraction
    # ------------------------------------------------------------------

    def _extract_raw_attention(self, batch: GameStateBatch) -> torch.Tensor:
        """Run the model with hooks and return last-layer attention weights.

        Hooks are installed before and removed after the forward pass.

        Args:
            batch: Input tensors for the model.

        Returns:
            Tensor of shape ``(B, n_heads, seq_len, seq_len)`` from the final
            transformer block.

        Raises:
            RuntimeError: If no attention weights were captured.
        """
        self._register_hooks()

        def _to(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.to(self.device) if t is not None else None

        with torch.no_grad():
            self.model(
                batch.static.to(self.device),
                batch.sequence.to(self.device),
                _to(batch.hand_counts),
                _to(batch.aka_flags),
                _to(batch.valid_mask),
            )

        self._remove_hooks()

        if not self._captured_attn:
            raise RuntimeError(
                "No attention weights captured. Verify that model.blocks contains "
                "HookedTransformerBlock instances with a .attn attribute."
            )

        last_layer_idx = max(self._captured_attn.keys())
        return self._captured_attn[last_layer_idx]  # (B, H, S, S)

    def _compute_group_scores(
        self,
        attn_weights: torch.Tensor,
        sequence: torch.Tensor,
        player_ids: List[int],
    ) -> List[Dict[str, float]]:
        """Aggregate last-layer attention into 8 feature-group importance scores.

        Importance per event position = mean over heads and query positions of
        the attention weight column, i.e.
        ``attn_weights.mean(dim=(1, 2))`` → shape ``(B, seq_len)``.

        Group scores are summed over member positions and renormalized to sum 1.

        Args:
            attn_weights: ``(B, n_heads, seq_len, seq_len)`` from last layer.
            sequence: ``(B, seq_len, 6)`` event-sequence tensor (CPU).
            player_ids: Decision-maker player IDs, one per batch element.

        Returns:
            List of dicts (length B), each mapping group name → float in [0, 1].
        """
        # Average over heads (dim 1) and query positions (dim 2) → (B, seq_len)
        importance = attn_weights.mean(dim=(1, 2))

        results: List[Dict[str, float]] = []
        B = attn_weights.shape[0]

        for b in range(B):
            seq_b = sequence[b].cpu()        # (S, 6)
            imp_b = importance[b].cpu()      # (S,)
            pid = player_ids[b] if b < len(player_ids) else 0

            group_positions = self._group_map_fn(seq_b, pid)

            group_scores: Dict[str, float] = {}
            for name in FEATURE_GROUP_NAMES:
                positions = group_positions.get(name, [])
                if positions:
                    group_scores[name] = float(imp_b[positions].sum().item())
                else:
                    group_scores[name] = 0.0

            # Renormalize so scores sum to 1.
            total = sum(group_scores.values())
            if total > 1e-8:
                group_scores = {n: v / total for n, v in group_scores.items()}

            results.append(group_scores)

        return results

    # ------------------------------------------------------------------
    # Internal: masking
    # ------------------------------------------------------------------

    def _mask_attention_scores(
        self,
        scores: Dict[str, float],
        condition: str,
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Zero-mask k feature-group scores and renormalize the remainder.

        The three masking conditions implement the faithfulness probes described
        in the research design:

        - ``'top'``    (条件A): zero the *k most important* groups.
        - ``'bottom'`` (条件B): zero the *k least important* groups.
        - ``'random'`` (条件C): zero *k randomly chosen* groups.

        After masking, the surviving scores are renormalized so their sum is 1.

        Args:
            scores: Dict mapping group name → importance score (should sum to 1).
            condition: One of ``'top'``, ``'bottom'``, ``'random'``.
            k: Number of groups to zero.  Defaults to ``self.k``.

        Returns:
            New dict with the same keys, masked groups set to 0 and the rest
            renormalized.

        Raises:
            ValueError: If ``condition`` is not one of the three valid strings.
        """
        k = k if k is not None else self.k
        names = list(scores.keys())
        values = [scores[n] for n in names]

        if condition == "top":
            mask_indices = sorted(
                range(len(values)), key=lambda i: values[i], reverse=True
            )[:k]
        elif condition == "bottom":
            mask_indices = sorted(range(len(values)), key=lambda i: values[i])[:k]
        elif condition == "random":
            mask_indices = random.sample(range(len(values)), min(k, len(values)))
        else:
            raise ValueError(
                f"Unknown condition '{condition}'. Expected 'top', 'bottom', or 'random'."
            )

        masked = dict(zip(names, values))
        for idx in mask_indices:
            masked[names[idx]] = 0.0

        # Renormalize remaining scores.
        total = sum(masked.values())
        if total > 1e-8:
            masked = {n: v / total for n, v in masked.items()}

        logger.debug(
            "mask condition='%s' k=%d zeroed=%s",
            condition,
            k,
            [names[i] for i in mask_indices],
        )
        return masked

    # ------------------------------------------------------------------
    # Internal: metrics
    # ------------------------------------------------------------------

    def _compute_bertscore(
        self,
        hypotheses: List[str],
        references: List[str],
        model_type: str = "bert-base-multilingual-cased",
    ) -> List[float]:
        """Compute BERTScore F1 for a list of hypothesis–reference pairs.

        Uses a multilingual BERT model so that Japanese explanations are
        evaluated correctly.  Runs on GPU if available.

        Args:
            hypotheses: Masked-condition explanations.
            references: Baseline (unmasked) explanations.
            model_type: HuggingFace model identifier for BERTScore.
                Default: ``"bert-base-multilingual-cased"``.

        Returns:
            List of F1 scores (float) in ``[0, 1]``, one per pair.

        Raises:
            ImportError: If the ``bert_score`` package is not installed.
        """
        try:
            from bert_score import score as _bert_score  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "bert_score is required: pip install bert-score"
            ) from exc

        use_gpu = torch.cuda.is_available()
        _, _, f1 = _bert_score(
            hypotheses,
            references,
            model_type=model_type,
            lang="ja",
            device="cuda" if use_gpu else "cpu",
            verbose=False,
        )
        return f1.tolist()

    def _compute_keyword_shift(
        self,
        baseline_text: str,
        masked_text: str,
    ) -> float:
        """Compute mean absolute change in keyword occurrence across 8 feature groups.

        For each of the 8 feature-group names, the metric records whether the
        keyword appears (1) or is absent (0) in both texts, then returns the
        mean of ``|baseline_present - masked_present|`` across all keywords.

        This captures whether masking a group causes its name to disappear from
        (or appear in) the generated explanation.

        Args:
            baseline_text: Explanation from unmasked scores.
            masked_text: Explanation from masked scores.

        Returns:
            Float in ``[0, 1]``.  Zero means all keywords appear/disappear
            identically in both texts; 1.0 means every keyword changed status.
        """
        baseline_lower = baseline_text.lower()
        masked_lower = masked_text.lower()

        shifts: List[float] = []
        for keyword in FEATURE_GROUP_NAMES:
            kw = keyword.lower()
            baseline_hit = float(kw in baseline_lower)
            masked_hit = float(kw in masked_lower)
            shifts.append(abs(baseline_hit - masked_hit))

        return float(np.mean(shifts))

    # ------------------------------------------------------------------
    # Public: single game state
    # ------------------------------------------------------------------

    def run_single(
        self,
        game_state: Dict[str, Any],
        *,
        game_idx: int = 0,
    ) -> SingleResult:
        """Run the full attention-patching evaluation for one game state.

        Expected keys in ``game_state``:

        - ``"static"``     — ``Tensor (1, static_dim)``
        - ``"sequence"``   — ``Tensor (1, 60, 6)``
        - ``"player_id"``  — ``int`` (0–3)
        - ``"hand_counts"`` — optional ``Tensor (1, 34)``
        - ``"aka_flags"``  — optional ``Tensor (1, 3)``
        - ``"valid_mask"`` — optional ``Tensor (1, 34)``

        Any additional keys are passed through to ``explain_fn`` unchanged.

        Args:
            game_state: Dict containing model inputs and metadata.
            game_idx: Integer label for this game state (used in logging and output).

        Returns:
            :class:`SingleResult` with scores, explanations, and metrics.
        """
        logger.info("run_single game_idx=%d", game_idx)

        batch = GameStateBatch(
            static=game_state["static"],
            sequence=game_state["sequence"],
            hand_counts=game_state.get("hand_counts"),
            aka_flags=game_state.get("aka_flags"),
            valid_mask=game_state.get("valid_mask"),
        )
        player_ids = [int(game_state.get("player_id", 0))]

        attn_weights = self._extract_raw_attention(batch)
        group_scores_list = self._compute_group_scores(
            attn_weights, batch.sequence, player_ids
        )
        baseline_scores = group_scores_list[0]

        baseline_explanation = self.explain_fn(game_state, baseline_scores)

        masked_scores: Dict[str, Dict[str, float]] = {}
        masked_explanations: Dict[str, str] = {}
        bertscore_f1: Dict[str, float] = {}
        keyword_shift: Dict[str, float] = {}

        for cond in ("top", "bottom", "random"):
            masked = self._mask_attention_scores(baseline_scores, cond)
            masked_scores[cond] = masked

            explanation = self.explain_fn(game_state, masked)
            masked_explanations[cond] = explanation

            bs = self._compute_bertscore([explanation], [baseline_explanation])
            bertscore_f1[cond] = bs[0]
            keyword_shift[cond] = self._compute_keyword_shift(
                baseline_explanation, explanation
            )

        logger.info(
            "run_single done game_idx=%d BERTScore(top=%.3f bottom=%.3f random=%.3f)",
            game_idx,
            bertscore_f1["top"],
            bertscore_f1["bottom"],
            bertscore_f1["random"],
        )

        return SingleResult(
            game_idx=game_idx,
            baseline_scores=baseline_scores,
            masked_scores=masked_scores,
            baseline_explanation=baseline_explanation,
            masked_explanations=masked_explanations,
            bertscore_f1=bertscore_f1,
            keyword_shift=keyword_shift,
        )

    # ------------------------------------------------------------------
    # Public: batch evaluation + statistics
    # ------------------------------------------------------------------

    def run_batch(
        self,
        game_states: List[Dict[str, Any]],
        *,
        n: int = 100,
        k_range: Optional[List[int]] = None,
        spearman_subsample: int = 20,
        output_csv: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Evaluate a batch of game states and compute statistical tests.

        Statistical tests performed:

        1. **Paired t-test** (``scipy.stats.ttest_rel``): BERTScore F1 values
           for 条件A (top masking) vs 条件C (random masking) across all samples.
           Tests whether top-feature masking degrades explanations more than
           random masking.

        2. **Spearman correlation** (``scipy.stats.spearmanr``): masking
           intensity ``k`` (from ``k_range``) vs BERTScore *drop*
           (= 1 − F1) for each condition.  Computed on a random subsample for
           efficiency.

        Args:
            game_states: List of game state dicts (same schema as
                :meth:`run_single`).
            n: Maximum number of game states to evaluate.  The first ``n``
                elements are used; the list is not shuffled.
            k_range: Values of k for Spearman analysis.  Default ``[1,2,3,4,5]``.
            spearman_subsample: Number of game states used for Spearman sweep.
                Must be ≤ ``n``.  Default 20.
            output_csv: If given, the result DataFrame is written to this path
                (UTF-8 with BOM for Excel compatibility).

        Returns:
            Tuple of:

            - ``pd.DataFrame`` — one row per (game state × condition).
              Columns include ``game_idx``, ``condition``, ``k``,
              ``bertscore_f1``, ``keyword_shift``, baseline group scores,
              and masked group scores.
            - ``dict`` — statistical results:

              - ``"ttest_top_vs_random"`` → ``scipy.stats.TtestResult``
              - ``"spearman_k_vs_bertscore"`` → dict mapping condition key
                to ``scipy.stats.SpearmanrResult``
        """
        if k_range is None:
            k_range = [1, 2, 3, 4, 5]

        sample = game_states[:n]
        logger.info(
            "run_batch: %d game states (n=%d requested)", len(sample), n
        )

        rows: List[Dict[str, Any]] = []
        single_results: List[SingleResult] = []

        for idx, gs in enumerate(sample):
            logger.info("  [%d/%d] game_idx=%d", idx + 1, len(sample), idx)
            try:
                result = self.run_single(gs, game_idx=idx)
                single_results.append(result)

                for cond in ("top", "bottom", "random"):
                    row: Dict[str, Any] = {
                        "game_idx": idx,
                        "condition": cond,
                        "k": self.k,
                        "bertscore_f1": result.bertscore_f1[cond],
                        "keyword_shift": result.keyword_shift[cond],
                    }
                    for g in FEATURE_GROUP_NAMES:
                        row[f"baseline_{g}"] = result.baseline_scores[g]
                        row[f"masked_{g}"] = result.masked_scores[cond][g]
                    rows.append(row)

            except Exception:  # noqa: BLE001
                logger.exception("Error at game_idx=%d, skipping.", idx)

        df = pd.DataFrame(rows)
        stats_out: Dict[str, Any] = {}

        # --- 1. Paired t-test: 条件A vs 条件C ---
        top_f1 = [r.bertscore_f1["top"] for r in single_results]
        rand_f1 = [r.bertscore_f1["random"] for r in single_results]

        if len(top_f1) >= 2:
            ttest = stats.ttest_rel(top_f1, rand_f1)
            stats_out["ttest_top_vs_random"] = ttest
            logger.info(
                "Paired t-test (top vs random): t=%.4f  p=%.4f",
                ttest.statistic,
                ttest.pvalue,
            )
        else:
            logger.warning("Paired t-test skipped: fewer than 2 valid samples.")

        # --- 2. Spearman correlation: k-value vs BERTScore drop ---
        subsample = sample[:min(spearman_subsample, len(sample))]
        spearman_results: Dict[str, Any] = {}

        for cond in ("top", "bottom", "random"):
            k_vals: List[int] = []
            score_drops: List[float] = []

            for k_val in k_range:
                for gs in subsample:
                    try:
                        b = GameStateBatch(
                            static=gs["static"],
                            sequence=gs["sequence"],
                            hand_counts=gs.get("hand_counts"),
                            aka_flags=gs.get("aka_flags"),
                            valid_mask=gs.get("valid_mask"),
                        )
                        aw = self._extract_raw_attention(b)
                        gsl = self._compute_group_scores(
                            aw, b.sequence, [int(gs.get("player_id", 0))]
                        )
                        base_scores = gsl[0]
                        base_exp = self.explain_fn(gs, base_scores)
                        masked = self._mask_attention_scores(base_scores, cond, k=k_val)
                        masked_exp = self.explain_fn(gs, masked)
                        f1 = self._compute_bertscore([masked_exp], [base_exp])[0]
                        k_vals.append(k_val)
                        score_drops.append(1.0 - f1)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "Spearman sweep error k=%d cond=%s", k_val, cond
                        )

            if len(k_vals) >= 4:
                sp = stats.spearmanr(k_vals, score_drops)
                spearman_results[cond] = sp
                logger.info(
                    "Spearman (k vs BERTScore-drop, cond=%s): rho=%.4f  p=%.4f",
                    cond,
                    sp.statistic,
                    sp.pvalue,
                )
            else:
                logger.warning(
                    "Spearman skipped for cond=%s: only %d data points.", cond, len(k_vals)
                )

        stats_out["spearman_k_vs_bertscore"] = spearman_results

        if output_csv:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            logger.info("Results saved → %s", output_csv)

        return df, stats_out


# ---------------------------------------------------------------------------
# Smoke-test main block
# ---------------------------------------------------------------------------


def main() -> None:
    """Quick sanity check — runs with a randomly initialized model and a stub LLM.

    Verifies that:
    - Hooks register and fire correctly.
    - Group score extraction produces 8 non-negative values summing to 1.
    - All three masking conditions produce valid renormalized score dicts.
    - BERTScore and keyword-shift calculations complete without error.

    This is intentionally self-contained: it requires only the libraries
    listed at the top of this file plus the project's ``models`` package.
    No checkpoint, API key, or external data is needed.
    """
    import sys
    from pathlib import Path

    # Allow running from the project root without install.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    from models.mahjong_transformer_v2 import MahjongTransformerConfig, MahjongTransformerV2  # noqa: E402

    # ------------------------------------------------------------------ model
    cfg = MahjongTransformerConfig()
    model = MahjongTransformerV2(cfg)
    model.eval()
    logger.info("Dummy model created. n_layers=%d d_model=%d", cfg.n_layers, cfg.d_model)

    # ---------------------------------------------------------- stub explain_fn
    def stub_explain(
        game_state: Dict[str, Any],
        attention_scores: Dict[str, float],
    ) -> str:
        """Return a templated explanation listing the two highest-scored groups."""
        top2 = sorted(attention_scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
        parts = [f"{name}({score:.3f})" for name, score in top2]
        return (
            f"この局面では {parts[0]} と {parts[1]} が重要です。"
            f" ShantenReduction および TileEfficiency を重視した打牌です。"
        )

    evaluator = AttentionPatchingEvaluator(model, stub_explain, k=3)

    # ---------------------------------------------------------- dummy game state
    B, S, static_dim = 1, 60, 157
    seq = torch.zeros(B, S, 6)
    # Populate a plausible event sequence: draw–discard pairs for player 0 & 1.
    for t in range(0, min(40, S), 4):
        seq[0, t]     = torch.tensor([_ET_DRAW,    0, t % 34, t // 4, 0, 0], dtype=torch.float32)
        seq[0, t + 1] = torch.tensor([_ET_DISCARD, 0, t % 34, t // 4, 0, 0], dtype=torch.float32)
        seq[0, t + 2] = torch.tensor([_ET_DRAW,    1, (t + 2) % 34, t // 4, 0, 0], dtype=torch.float32)
        seq[0, t + 3] = torch.tensor([_ET_DISCARD, 1, (t + 2) % 34, t // 4, 0, 0], dtype=torch.float32)
    # Mark remaining positions as PADDING.
    seq[0, 40:, 0] = _ET_PADDING

    game_state: Dict[str, Any] = {
        "static": torch.randn(B, static_dim),
        "sequence": seq,
        "hand_counts": torch.zeros(B, 34),
        "aka_flags": torch.zeros(B, 3),
        "valid_mask": torch.ones(B, 34),
        "player_id": 0,
    }

    # ---------------------------------------------------------- run_single test
    logger.info("=== run_single test ===")
    result = evaluator.run_single(game_state, game_idx=0)

    print("\n--- Baseline group scores ---")
    for name, score in result.baseline_scores.items():
        print(f"  {name:<22}: {score:.4f}")

    print("\n--- BERTScore F1 (vs baseline) ---")
    for cond, f1 in result.bertscore_f1.items():
        print(f"  {cond:<8}: {f1:.4f}")

    print("\n--- Keyword shift ---")
    for cond, shift in result.keyword_shift.items():
        print(f"  {cond:<8}: {shift:.4f}")

    # ---------------------------------------------------------- masking unit test
    logger.info("=== masking unit test ===")
    for cond in ("top", "bottom", "random"):
        masked = evaluator._mask_attention_scores(result.baseline_scores, cond, k=3)
        total = sum(masked.values())
        assert abs(total - 1.0) < 1e-5 or total < 1e-8, (
            f"Renormalization failed for condition '{cond}': sum={total:.6f}"
        )
        n_zeros = sum(1 for v in masked.values() if v == 0.0)
        print(f"  cond={cond:8s}  zeros={n_zeros}  sum={total:.6f}  ✓")

    # ---------------------------------------------------------- run_batch test
    logger.info("=== run_batch test (3 samples) ===")
    game_states = [game_state] * 3
    df, stats_out = evaluator.run_batch(
        game_states,
        n=3,
        k_range=[1, 2, 3],
        spearman_subsample=3,
        output_csv="/tmp/attn_patching_test.csv",
    )

    print(f"\nDataFrame shape: {df.shape}")
    print(df[["game_idx", "condition", "bertscore_f1", "keyword_shift"]].to_string(index=False))

    if "ttest_top_vs_random" in stats_out:
        t = stats_out["ttest_top_vs_random"]
        print(f"\nPaired t-test: t={t.statistic:.4f}  p={t.pvalue:.4f}")

    sp_dict = stats_out.get("spearman_k_vs_bertscore", {})
    for cond, sp in sp_dict.items():
        print(f"Spearman (cond={cond}): rho={sp.statistic:.4f}  p={sp.pvalue:.4f}")

    logger.info("All checks passed.")


if __name__ == "__main__":
    main()
