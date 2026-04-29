#!/usr/bin/env python3
"""Export model discard decisions as a static visualization report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.observation_schema import (  # noqa: E402
    EVENT_TYPES,
    MAX_EVENT_HISTORY,
    EventRecord,
    ObservedState,
    PrivateRoundState,
    PublicDiscard,
    PublicMeld,
    _parse_discard_tag,
    _parse_draw_tag,
    iter_xml_files,
    row_from_observation,
)
from models.mahjong_transformer_v2 import MahjongTransformerConfig, MahjongTransformerV2  # noqa: E402
from utils.tile_utils import (  # noqa: E402
    NUM_TILE_TYPES,
    is_aka_dora,
    tile_id_to_kind,
    tile_id_to_string,
    tile_kind_to_id,
    tile_kind_to_string,
)
from utils.xml_parser import GameMeta, RoundData, parse_tenhou_xml  # noqa: E402


WINDS = ["東", "南", "西", "北"]
PLAYER_LABELS = ["自家", "下家", "対面", "上家"]
EVENT_TYPE_LABELS = {value: key for key, value in EVENT_TYPES.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Tenhou XML file or directory")
    parser.add_argument("--checkpoint", required=True, help="MahjongTransformerV2 checkpoint")
    parser.add_argument("--output", required=True, help="Output report JSON path")
    parser.add_argument("--limit-files", type=int, default=5)
    parser.add_argument("--offset-files", type=int, default=0)
    parser.add_argument("--limit-decisions", type=int, default=200)
    parser.add_argument("--limit-frames", type=int, default=1000, help="Number of replay events to export for the viewer")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10, help="Number of candidates to emphasize in the UI")
    parser.add_argument("--include-call-discards", action="store_true", help="Include discards after calls")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def tile_code_for_kind(kind: int) -> str:
    if 0 <= kind <= 8:
        return f"{kind + 1}m"
    if 9 <= kind <= 17:
        return f"{kind - 8}p"
    if 18 <= kind <= 26:
        return f"{kind - 17}s"
    honor_codes = ["east", "south", "west", "north", "white", "green", "red"]
    if 27 <= kind <= 33:
        return honor_codes[kind - 27]
    return "unknown"


def tile_code_for_id(tile_id: int) -> str:
    if is_aka_dora(tile_id):
        return {16: "0m", 52: "0p", 88: "0s"}.get(tile_id, "unknown")
    return tile_code_for_kind(tile_id_to_kind(tile_id))


def tile_payload(tile_id: int) -> Dict[str, Any]:
    kind = tile_id_to_kind(tile_id)
    return {
        "id": int(tile_id),
        "kind": int(kind),
        "name": tile_id_to_string(tile_id),
        "code": tile_code_for_id(tile_id),
        "is_aka": bool(is_aka_dora(tile_id)),
    }


def action_tile_payload(kind: int) -> Dict[str, Any]:
    tile_id = tile_kind_to_id(kind)
    return {
        "kind": int(kind),
        "name": tile_kind_to_string(kind),
        "code": tile_code_for_kind(kind),
        "representative_id": int(tile_id),
    }


def discard_payload(discard: PublicDiscard) -> Dict[str, Any]:
    payload = tile_payload(discard.tile_id)
    payload.update(
        {
            "player": int(discard.player),
            "tsumogiri": bool(discard.tsumogiri),
            "called": bool(discard.called),
        }
    )
    return payload


def meld_payload(meld: PublicMeld) -> Dict[str, Any]:
    return {
        "player": int(meld.player),
        "type": meld.naki_type,
        "tiles": [tile_payload(tile_id) for tile_id in meld.tiles],
        "consumed": [tile_payload(tile_id) for tile_id in meld.consumed],
        "called_tile": tile_payload(meld.called_tile) if meld.called_tile >= 0 else None,
        "from_who": int(meld.from_who),
    }


def event_payload(event: EventRecord, index: int, actor: int) -> Dict[str, Any]:
    tile_id = event.tile_id
    hidden = bool(event.private_tile and event.player != actor)
    if hidden:
        tile_id = -1
    return {
        "index": int(index),
        "type": event.event_type,
        "player": int(event.player),
        "tile": tile_payload(tile_id) if tile_id >= 0 else None,
        "hidden": hidden,
        "junme": float(event.junme),
        "data0": float(event.data0),
        "data1": float(event.data1),
        "is_padding": False,
    }


def sequence_event_payloads(observed: ObservedState) -> List[Dict[str, Any]]:
    payloads = [
        event_payload(event, index, observed.player_id)
        for index, event in enumerate(observed.observable_events()[-MAX_EVENT_HISTORY:])
    ]
    for index in range(len(payloads), MAX_EVENT_HISTORY):
        payloads.append(
            {
                "index": int(index),
                "type": "PADDING",
                "player": -1,
                "tile": None,
                "hidden": False,
                "junme": 0.0,
                "data0": 0.0,
                "data1": 0.0,
                "is_padding": True,
            }
        )
    return payloads


def observed_payload(observed: ObservedState, meta: GameMeta) -> Dict[str, Any]:
    player_id = observed.player_id
    relative_players = []
    for offset in range(4):
        abs_player = (player_id + offset) % 4
        relative_players.append(
            {
                "seat": int(abs_player),
                "relative": int(offset),
                "label": PLAYER_LABELS[offset],
                "name": meta.player_names[abs_player] or f"player_{abs_player}",
                "score": int(observed.scores[abs_player]),
                "is_dealer": bool(abs_player == observed.dealer),
                "is_actor": bool(abs_player == player_id),
                "reach_status": int(observed.reach_status[abs_player]),
                "reach_junme": float(observed.reach_junme[abs_player]),
                "river": [discard_payload(discard) for discard in observed.rivers[abs_player]],
                "melds": [meld_payload(meld) for meld in observed.melds[abs_player]],
            }
        )

    return {
        "actor": int(player_id),
        "round": {
            "wind": int(observed.round_wind),
            "wind_label": WINDS[observed.round_wind] if 0 <= observed.round_wind < len(WINDS) else "?",
            "number": int(observed.round_num + 1),
            "honba": int(observed.honba),
            "kyotaku": int(observed.kyotaku),
            "dealer": int(observed.dealer),
            "junme": float(observed.junme),
            "wall_tile_count": int(observed.wall_tile_count),
        },
        "players": relative_players,
        "hand": [tile_payload(tile_id) for tile_id in observed.hand],
        "dora_indicators": [tile_payload(tile_id) for tile_id in observed.dora_indicators],
        "valid_action_kinds": [int(i) for i, value in enumerate(observed.valid_mask()) if value > 0],
        "last_draw_tile": tile_payload(observed.last_draw_tile) if observed.last_draw_tile >= 0 else None,
        "sequence_events": sequence_event_payloads(observed),
    }


def list_selected_files(path: Path, offset: int, limit: Optional[int]) -> List[Path]:
    files = iter_xml_files(path)
    if offset:
        files = files[offset:]
    if limit:
        files = files[:limit]
    return files


def load_model(checkpoint_path: Path, device: str) -> MahjongTransformerV2:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MahjongTransformerV2(MahjongTransformerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def iter_decision_examples(
    files: Iterable[Path],
    *,
    include_call_discards: bool,
    limit_decisions: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    summary = {
        "files_seen": 0,
        "files_processed": 0,
        "rounds_processed": 0,
        "decisions_exported": 0,
        "skipped_sanma": 0,
        "skipped_invalid_label": 0,
        "errors": [],
    }

    for xml_file in files:
        if len(examples) >= limit_decisions:
            break
        summary["files_seen"] += 1
        try:
            meta, rounds = parse_tenhou_xml(str(xml_file))
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"{xml_file.name}: {exc}")
            continue

        summary["files_processed"] += 1
        if meta.is_sanma:
            summary["skipped_sanma"] += 1
            continue

        for round_data in rounds:
            if len(examples) >= limit_decisions:
                break
            summary["rounds_processed"] += 1
            examples.extend(
                extract_round_examples(
                    xml_file,
                    meta,
                    round_data,
                    include_call_discards=include_call_discards,
                    remaining=limit_decisions - len(examples),
                    summary=summary,
                )
            )

    summary["decisions_exported"] = len(examples)
    return examples, summary


def iter_replay_frames(
    files: Iterable[Path],
    *,
    limit_frames: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    summary = {
        "frame_steps_exported": 0,
        "frames_exported": 0,
        "files_seen": 0,
        "files_processed": 0,
        "rounds_processed": 0,
        "skipped_sanma": 0,
        "errors": [],
    }

    for xml_file in files:
        if summary["frame_steps_exported"] >= limit_frames:
            break
        summary["files_seen"] += 1
        try:
            meta, rounds = parse_tenhou_xml(str(xml_file))
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"{xml_file.name}: {exc}")
            continue

        summary["files_processed"] += 1
        if meta.is_sanma:
            summary["skipped_sanma"] += 1
            continue

        for round_data in rounds:
            if summary["frame_steps_exported"] >= limit_frames:
                break
            summary["rounds_processed"] += 1
            round_frames, steps = extract_round_frames(
                xml_file,
                meta,
                round_data,
                remaining_steps=limit_frames - summary["frame_steps_exported"],
            )
            frames.extend(round_frames)
            summary["frame_steps_exported"] += steps

    summary["frames_exported"] = len(frames)
    return frames, summary


def extract_round_examples(
    xml_file: Path,
    meta: GameMeta,
    round_data: RoundData,
    *,
    include_call_discards: bool,
    remaining: int,
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    state = PrivateRoundState.from_round(round_data)

    for event_index, event in enumerate(round_data.events):
        if len(examples) >= remaining:
            break

        tag = event["tag"]
        attrib = event.get("attrib", {})

        draw = _parse_draw_tag(tag)
        if draw is not None:
            player, tile_id = draw
            state.process_draw(player, tile_id)
            continue

        discard = _parse_discard_tag(tag)
        if discard is not None:
            player, tile_id, tsumogiri = discard
            if state.pending_player == player and (include_call_discards or state.pending_source == "draw"):
                observed = state.observe(player)
                metadata = {
                    "source_file": xml_file.name,
                    "round_index": round_data.round_index,
                    "event_index": event_index,
                    "player_id": player,
                    "junme": state.junme,
                    "decision_source": state.pending_source,
                    "is_tsumogiri": bool(tsumogiri),
                    "label_tile_id": tile_id,
                }
                row = row_from_observation(observed, tile_id, metadata)
                if row is None:
                    summary["skipped_invalid_label"] += 1
                else:
                    examples.append(
                        {
                            "row": row,
                            "metadata": metadata,
                            "board": observed_payload(observed, meta),
                        }
                    )
            state.process_discard(player, tile_id, tsumogiri)
            continue

        if tag == "N":
            state.process_naki(int(attrib.get("who", -1)), int(attrib.get("m", 0)))
        elif tag == "REACH":
            state.process_reach(int(attrib.get("who", -1)), int(attrib.get("step", 0)))
        elif tag == "DORA":
            state.process_dora(int(attrib.get("hai", -1)))
        elif tag in {"AGARI", "RYUUKYOKU"}:
            state.process_terminal(tag, attrib)

    return examples


def extract_round_frames(
    xml_file: Path,
    meta: GameMeta,
    round_data: RoundData,
    *,
    remaining_steps: int,
) -> tuple[List[Dict[str, Any]], int]:
    frames: List[Dict[str, Any]] = []
    steps = 0
    state = PrivateRoundState.from_round(round_data)

    def append_viewer_frames(event_index: int, event_type: str, event_player: int) -> None:
        latest_event = state.event_history[-1] if state.event_history else EventRecord(event_type, player=event_player)
        for viewer in range(4):
            observed = state.observe(viewer)
            frames.append(
                {
                    "metadata": {
                        "source_file": xml_file.name,
                        "round_index": round_data.round_index,
                        "event_index": event_index,
                        "player_id": viewer,
                        "event_type": event_type,
                        "event_player": event_player,
                    },
                    "current_event": event_payload(latest_event, event_index, viewer),
                    "board": observed_payload(observed, meta),
                }
            )

    for event_index, event in enumerate(round_data.events):
        if steps >= remaining_steps:
            break

        tag = event["tag"]
        attrib = event.get("attrib", {})
        event_type = ""
        event_player = -1

        draw = _parse_draw_tag(tag)
        if draw is not None:
            event_player, tile_id = draw
            state.process_draw(event_player, tile_id)
            event_type = "DRAW"
        else:
            discard = _parse_discard_tag(tag)
            if discard is not None:
                event_player, tile_id, tsumogiri = discard
                state.process_discard(event_player, tile_id, tsumogiri)
                event_type = "DISCARD"
            elif tag == "N":
                event_player = int(attrib.get("who", -1))
                state.process_naki(event_player, int(attrib.get("m", 0)))
                event_type = "N"
            elif tag == "REACH":
                event_player = int(attrib.get("who", -1))
                state.process_reach(event_player, int(attrib.get("step", 0)))
                event_type = "REACH"
            elif tag == "DORA":
                state.process_dora(int(attrib.get("hai", -1)))
                event_type = "DORA"
            elif tag in {"AGARI", "RYUUKYOKU"}:
                event_player = int(attrib.get("who", -1))
                state.process_terminal(tag, attrib)
                event_type = "AGARI" if tag == "AGARI" else "RYUUKYOKU"

        if event_type:
            append_viewer_frames(event_index, event_type, event_player)
            steps += 1

    return frames, steps


@torch.no_grad()
def attach_predictions(
    examples: List[Dict[str, Any]],
    model: MahjongTransformerV2,
    *,
    batch_size: int,
    top_k: int,
    device: str,
) -> None:
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        rows = [example["row"] for example in batch]
        static = torch.tensor(np.stack([row.static_features for row in rows]), dtype=torch.float32, device=device)
        sequence = torch.tensor(np.stack([row.sequence_features for row in rows]), dtype=torch.float32, device=device)
        hand_counts = torch.tensor(np.stack([row.hand_counts for row in rows]), dtype=torch.float32, device=device)
        aka_flags = torch.tensor(np.stack([row.aka_flags for row in rows]), dtype=torch.float32, device=device)
        valid_mask = torch.tensor(np.stack([row.valid_mask for row in rows]), dtype=torch.float32, device=device)

        logits, internals = model(static, sequence, hand_counts, aka_flags, valid_mask, return_internals=True)
        probs = torch.softmax(logits, dim=-1)
        attention = internals["attn_weights"][-1].mean(dim=(1, 2)).detach().cpu().numpy() if internals["attn_weights"] else None

        for offset, row in enumerate(rows):
            row_logits = logits[offset].detach().cpu().numpy()
            row_probs = probs[offset].detach().cpu().numpy()
            candidates = []
            for kind in range(NUM_TILE_TYPES):
                if row.valid_mask[kind] <= 0:
                    continue
                candidates.append(
                    {
                        "tile": action_tile_payload(kind),
                        "logit": float(row_logits[kind]),
                        "probability": float(row_probs[kind]),
                        "is_actual": bool(kind == row.label),
                    }
                )
            candidates.sort(key=lambda item: item["probability"], reverse=True)
            for rank, candidate in enumerate(candidates, start=1):
                candidate["rank"] = rank

            actual = next((candidate for candidate in candidates if candidate["is_actual"]), None)
            prediction = {
                "top_k": int(top_k),
                "predicted": candidates[0] if candidates else None,
                "actual": actual,
                "actual_rank": int(actual["rank"]) if actual else None,
                "candidates": candidates,
                "top_candidates": candidates[:top_k],
                "probability_sum": float(sum(candidate["probability"] for candidate in candidates)),
            }
            batch[offset]["prediction"] = prediction
            if attention is not None:
                batch[offset]["attention"] = attention_payload(batch[offset]["board"]["sequence_events"], attention[offset])
            del batch[offset]["row"]


def event_summary(event: Dict[str, Any]) -> str:
    event_type = event["type"]
    player = event["player"]
    player_text = f"P{player}" if player >= 0 else "-"
    tile = event.get("tile")
    tile_text = tile["name"] if tile else ("非公開" if event.get("hidden") else "")
    if event_type == "PADDING":
        return "padding"
    return " ".join(part for part in (event_type, player_text, tile_text) if part)


def attention_payload(events: List[Dict[str, Any]], weights: np.ndarray, top_n: int = 12) -> Dict[str, Any]:
    total = float(np.sum(weights)) or 1.0
    per_event = []
    for event, weight in zip(events, weights.tolist()):
        normalized = float(weight) / total
        item = {
            "index": int(event["index"]),
            "weight": float(weight),
            "normalized": normalized,
            "summary": event_summary(event),
            "event": event,
        }
        per_event.append(item)
    visible = [item for item in per_event if not item["event"].get("is_padding")]
    visible.sort(key=lambda item: item["weight"], reverse=True)
    return {
        "description": "Last-layer attention averaged over heads and query positions.",
        "top_events": visible[:top_n],
        "per_event": per_event,
    }


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input)
    files = list_selected_files(input_path, args.offset_files, args.limit_files)
    if not files:
        raise SystemExit("No XML files selected")

    model = load_model(Path(args.checkpoint), args.device)
    examples, summary = iter_decision_examples(
        files,
        include_call_discards=args.include_call_discards,
        limit_decisions=args.limit_decisions,
    )
    attach_predictions(examples, model, batch_size=args.batch_size, top_k=args.top_k, device=args.device)
    frames, frame_summary = iter_replay_frames(files, limit_frames=args.limit_frames)

    return {
        "schema": "mahjong_ai_decision_report_v1",
        "source": {
            "input": str(input_path),
            "checkpoint": str(Path(args.checkpoint)),
            "offset_files": int(args.offset_files),
            "limit_files": int(args.limit_files) if args.limit_files is not None else None,
            "limit_frames": int(args.limit_frames),
            "include_call_discards": bool(args.include_call_discards),
        },
        "summary": summary,
        "frame_summary": frame_summary,
        "frames": frames,
        "decisions": examples,
    }


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
