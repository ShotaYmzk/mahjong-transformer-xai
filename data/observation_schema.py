"""Leak-safe Tenhou XML observation extraction.

This module intentionally separates complete private replay state from model
inputs. `PrivateRoundState` may hold all hands so it can validate Tenhou event
transitions, but only `ObservedState` and `DatasetRow` are serialized.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from utils.naki_utils import NakiType, decode_naki
from utils.tile_utils import (
    AKA_MAN_ID,
    AKA_PIN_ID,
    AKA_SOU_ID,
    NUM_TILE_TYPES,
    is_valid_tile_id,
    tile_id_to_kind,
)
from utils.xml_parser import RoundData, parse_tenhou_xml


NUM_PLAYERS = 4
MAX_EVENT_HISTORY = 60
EVENT_FEATURE_DIM = 6
STATIC_FEATURE_DIM = 157

EVENT_TYPES = {
    "INIT": 0,
    "DRAW": 1,
    "DISCARD": 2,
    "N": 3,
    "REACH": 4,
    "DORA": 5,
    "AGARI": 6,
    "RYUUKYOKU": 7,
    "PADDING": 8,
}

DRAW_TAGS = {"T": 0, "U": 1, "V": 2, "W": 3}
DISCARD_TAGS = {"D": 0, "E": 1, "F": 2, "G": 3}
NAKI_TYPE_TO_CODE = {
    NakiType.CHI: 0,
    NakiType.PON: 1,
    NakiType.DAIMINKAN: 2,
    NakiType.KAKAN: 3,
    NakiType.ANKAN: 4,
    NakiType.UNKNOWN: -1,
}


@dataclass(frozen=True)
class PublicDiscard:
    player: int
    tile_id: int
    tsumogiri: bool
    called: bool = False


@dataclass(frozen=True)
class PublicMeld:
    player: int
    naki_type: str
    tiles: Tuple[int, ...]
    consumed: Tuple[int, ...]
    called_tile: int
    from_who: int
    raw_m: int


@dataclass(frozen=True)
class EventRecord:
    event_type: str
    player: int = -1
    tile_id: int = -1
    junme: float = 0.0
    data0: float = 0.0
    data1: float = 0.0
    private_tile: bool = False


@dataclass(frozen=True)
class ObservedState:
    """Only information visible to `player_id` at the decision point."""

    player_id: int
    hand: Tuple[int, ...]
    round_wind: int
    round_num: int
    honba: int
    kyotaku: int
    dealer: int
    scores: Tuple[int, int, int, int]
    dora_indicators: Tuple[int, ...]
    rivers: Tuple[Tuple[PublicDiscard, ...], ...]
    melds: Tuple[Tuple[PublicMeld, ...], ...]
    reach_status: Tuple[int, int, int, int]
    reach_junme: Tuple[float, float, float, float]
    wall_tile_count: int
    junme: float
    events: Tuple[EventRecord, ...]
    last_draw_tile: int = -1

    def hand_counts(self) -> np.ndarray:
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for tile_id in self.hand:
            kind = tile_id_to_kind(tile_id)
            if 0 <= kind < NUM_TILE_TYPES:
                counts[kind] += 1.0
        return counts

    def aka_flags(self) -> np.ndarray:
        return np.array(
            [
                float(AKA_MAN_ID in self.hand),
                float(AKA_PIN_ID in self.hand),
                float(AKA_SOU_ID in self.hand),
            ],
            dtype=np.float32,
        )

    def valid_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        if self.reach_status[self.player_id] == 2 and self.last_draw_tile >= 0:
            kind = tile_id_to_kind(self.last_draw_tile)
            if 0 <= kind < NUM_TILE_TYPES:
                mask[kind] = 1.0
            return mask

        for tile_id in self.hand:
            kind = tile_id_to_kind(tile_id)
            if 0 <= kind < NUM_TILE_TYPES:
                mask[kind] = 1.0
        return mask

    def static_features(self) -> np.ndarray:
        features = np.zeros(STATIC_FEATURE_DIM, dtype=np.float32)
        idx = 0

        # 8 context features
        features[idx : idx + 8] = np.array(
            [
                self.round_wind,
                self.honba,
                self.kyotaku,
                self.dealer,
                self.wall_tile_count,
                float(self.player_id == self.dealer),
                self.junme,
                len(self.dora_indicators),
            ],
            dtype=np.float32,
        )
        idx += 8

        # 5 own player features
        features[idx : idx + 5] = np.array(
            [
                self.reach_status[self.player_id],
                self.reach_junme[self.player_id],
                len(self.rivers[self.player_id]),
                len(self.melds[self.player_id]),
                len(self.hand),
            ],
            dtype=np.float32,
        )
        idx += 5

        # 34 own hand counts
        features[idx : idx + NUM_TILE_TYPES] = self.hand_counts()
        idx += NUM_TILE_TYPES

        # 34 dora indicator counts
        for tile_id in self.dora_indicators:
            kind = tile_id_to_kind(tile_id)
            if 0 <= kind < NUM_TILE_TYPES:
                features[idx + kind] += 1.0
        idx += NUM_TILE_TYPES

        # 34 own river counts
        for discard in self.rivers[self.player_id]:
            kind = tile_id_to_kind(discard.tile_id)
            if 0 <= kind < NUM_TILE_TYPES:
                features[idx + kind] += 1.0
        idx += NUM_TILE_TYPES

        # 34 visible tile counts: rivers, dora indicators, and exposed meld tiles.
        visible = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for river in self.rivers:
            for discard in river:
                kind = tile_id_to_kind(discard.tile_id)
                if 0 <= kind < NUM_TILE_TYPES:
                    visible[kind] += 1.0
        for tile_id in self.dora_indicators:
            kind = tile_id_to_kind(tile_id)
            if 0 <= kind < NUM_TILE_TYPES:
                visible[kind] += 1.0
        for player_melds in self.melds:
            for meld in player_melds:
                # The called tile is already visible as a discard; count consumed
                # exposed tiles here to avoid double-counting the called tile.
                for tile_id in meld.consumed:
                    kind = tile_id_to_kind(tile_id)
                    if 0 <= kind < NUM_TILE_TYPES:
                        visible[kind] += 1.0
        features[idx : idx + NUM_TILE_TYPES] = visible
        idx += NUM_TILE_TYPES

        # 8 relative position and reach features
        for offset in range(NUM_PLAYERS):
            abs_player = (self.player_id + offset) % NUM_PLAYERS
            features[idx + offset * 2] = float(abs_player == self.player_id)
            features[idx + offset * 2 + 1] = float(self.reach_status[abs_player] == 2)
        idx += 8

        if idx != STATIC_FEATURE_DIM:
            raise ValueError(f"static feature size mismatch: {idx}")
        return features

    def sequence_features(self) -> np.ndarray:
        rows: List[np.ndarray] = []
        for event in self.events[-MAX_EVENT_HISTORY:]:
            vec = np.zeros(EVENT_FEATURE_DIM, dtype=np.float32)
            vec[0] = float(EVENT_TYPES.get(event.event_type, EVENT_TYPES["PADDING"]))
            vec[1] = float(event.player + 1)
            tile_id = event.tile_id
            if event.private_tile and event.player != self.player_id:
                tile_id = -1
            kind = tile_id_to_kind(tile_id)
            vec[2] = float(kind + 1) if kind >= 0 else 0.0
            vec[3] = float(event.junme)
            vec[4] = float(event.data0)
            vec[5] = float(event.data1)
            rows.append(vec)

        padding = MAX_EVENT_HISTORY - len(rows)
        if padding > 0:
            pad = np.zeros(EVENT_FEATURE_DIM, dtype=np.float32)
            pad[0] = float(EVENT_TYPES["PADDING"])
            rows.extend([pad.copy() for _ in range(padding)])
        return np.stack(rows, axis=0).astype(np.float32)


@dataclass(frozen=True)
class DatasetRow:
    static_features: np.ndarray
    sequence_features: np.ndarray
    hand_counts: np.ndarray
    aka_flags: np.ndarray
    valid_mask: np.ndarray
    label: int
    metadata: Dict[str, Any]


@dataclass
class ExtractionReport:
    files_processed: int = 0
    rounds_processed: int = 0
    samples: int = 0
    draw_discards: int = 0
    call_discards: int = 0
    skipped_sanma: int = 0
    skipped_invalid_label: int = 0
    skipped_parse_errors: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_processed": self.files_processed,
            "rounds_processed": self.rounds_processed,
            "samples": self.samples,
            "draw_discards": self.draw_discards,
            "call_discards": self.call_discards,
            "skipped_sanma": self.skipped_sanma,
            "skipped_invalid_label": self.skipped_invalid_label,
            "skipped_parse_errors": self.skipped_parse_errors,
            "errors": list(self.errors),
        }


@dataclass
class PrivateRoundState:
    """Complete replay state. Do not serialize this object into datasets."""

    round_index: int = 0
    round_wind: int = 0
    round_num: int = 0
    honba: int = 0
    kyotaku: int = 0
    dealer: int = 0
    scores: List[int] = field(default_factory=lambda: [25000] * NUM_PLAYERS)
    private_hands: List[List[int]] = field(default_factory=lambda: [[] for _ in range(NUM_PLAYERS)])
    rivers: List[List[PublicDiscard]] = field(default_factory=lambda: [[] for _ in range(NUM_PLAYERS)])
    melds: List[List[PublicMeld]] = field(default_factory=lambda: [[] for _ in range(NUM_PLAYERS)])
    reach_status: List[int] = field(default_factory=lambda: [0] * NUM_PLAYERS)
    reach_junme: List[float] = field(default_factory=lambda: [-1.0] * NUM_PLAYERS)
    dora_indicators: List[int] = field(default_factory=list)
    event_history: Deque[EventRecord] = field(default_factory=lambda: deque(maxlen=MAX_EVENT_HISTORY))
    wall_tile_count: int = 70
    junme: float = 0.0
    last_draw_tile: List[int] = field(default_factory=lambda: [-1] * NUM_PLAYERS)
    last_discard_player: int = -1
    last_discard_tile: int = -1
    pending_player: int = -1
    pending_source: str = ""

    @classmethod
    def from_round(cls, round_data: RoundData) -> "PrivateRoundState":
        state = cls(
            round_index=round_data.round_index,
            round_wind=round_data.round_wind,
            round_num=round_data.round_num,
            honba=round_data.honba,
            kyotaku=round_data.kyotaku,
            dealer=round_data.dealer,
            scores=list(round_data.initial_scores),
            private_hands=[sorted(list(hand), key=lambda t: (tile_id_to_kind(t), t)) for hand in round_data.initial_hands],
            dora_indicators=[round_data.dora_indicator] if is_valid_tile_id(round_data.dora_indicator) else [],
        )
        state.wall_tile_count = 136 - 14 - sum(len(hand) for hand in state.private_hands)
        state.event_history.append(EventRecord("INIT", player=state.dealer, junme=0.0))
        if state.dora_indicators:
            state.event_history.append(EventRecord("DORA", player=-1, tile_id=state.dora_indicators[0], junme=0.0))
        return state

    def observe(self, player_id: int) -> ObservedState:
        return ObservedState(
            player_id=player_id,
            hand=tuple(self.private_hands[player_id]),
            round_wind=self.round_wind,
            round_num=self.round_num,
            honba=self.honba,
            kyotaku=self.kyotaku,
            dealer=self.dealer,
            scores=tuple(self.scores),  # type: ignore[arg-type]
            dora_indicators=tuple(self.dora_indicators),
            rivers=tuple(tuple(river) for river in self.rivers),
            melds=tuple(tuple(melds) for melds in self.melds),
            reach_status=tuple(self.reach_status),  # type: ignore[arg-type]
            reach_junme=tuple(self.reach_junme),  # type: ignore[arg-type]
            wall_tile_count=self.wall_tile_count,
            junme=self.junme,
            events=tuple(self.event_history),
            last_draw_tile=self.last_draw_tile[player_id],
        )

    def process_draw(self, player: int, tile_id: int) -> None:
        self.private_hands[player].append(tile_id)
        self.private_hands[player].sort(key=lambda t: (tile_id_to_kind(t), t))
        self.last_draw_tile[player] = tile_id
        self.pending_player = player
        self.pending_source = "draw"
        if self.wall_tile_count > 0:
            self.wall_tile_count -= 1
        if player == 0:
            self.junme = max(1.0, np.floor(self.junme) + 1.0)
        elif self.junme == 0.0:
            self.junme = 1.0
        self.event_history.append(EventRecord("DRAW", player=player, tile_id=tile_id, junme=self.junme, private_tile=True))

    def process_discard(self, player: int, tile_id: int, tsumogiri: bool) -> None:
        self._remove_tile_by_id_or_kind(player, tile_id)
        discard = PublicDiscard(player=player, tile_id=tile_id, tsumogiri=tsumogiri)
        self.rivers[player].append(discard)
        self.last_discard_player = player
        self.last_discard_tile = tile_id
        self.pending_player = -1
        self.pending_source = ""
        self.event_history.append(
            EventRecord("DISCARD", player=player, tile_id=tile_id, junme=self.junme, data0=float(tsumogiri))
        )
        if self.reach_status[player] == 1:
            self.reach_status[player] = 2
            self.reach_junme[player] = self.junme
            self.kyotaku += 1

    def process_reach(self, player: int, step: int) -> None:
        if step == 1 and self.reach_status[player] == 0:
            self.reach_status[player] = 1
        elif step == 2:
            self.reach_status[player] = 2
            self.reach_junme[player] = self.junme
        self.event_history.append(EventRecord("REACH", player=player, junme=self.junme, data0=float(step)))

    def process_dora(self, tile_id: int) -> None:
        if is_valid_tile_id(tile_id):
            self.dora_indicators.append(tile_id)
            self.event_history.append(EventRecord("DORA", player=-1, tile_id=tile_id, junme=self.junme))

    def process_naki(self, player: int, meld_code: int) -> None:
        info = decode_naki(meld_code)
        if info.naki_type == NakiType.UNKNOWN:
            return

        called_tile = self.last_discard_tile if info.naki_type in {NakiType.CHI, NakiType.PON, NakiType.DAIMINKAN} else -1
        from_who = self.last_discard_player if called_tile >= 0 else player
        consumed = self._resolve_consumed_tiles(player, info.consumed)
        for tile_id in consumed:
            self._remove_tile_by_id_or_kind(player, tile_id)

        if called_tile >= 0 and self.rivers[from_who]:
            last = self.rivers[from_who][-1]
            if last.tile_id == called_tile:
                self.rivers[from_who][-1] = PublicDiscard(last.player, last.tile_id, last.tsumogiri, called=True)

        tiles = tuple(sorted(([called_tile] if called_tile >= 0 else []) + list(consumed)))
        meld = PublicMeld(
            player=player,
            naki_type=info.naki_type.value,
            tiles=tiles,
            consumed=tuple(consumed),
            called_tile=called_tile,
            from_who=from_who,
            raw_m=meld_code,
        )
        self.melds[player].append(meld)
        self.event_history.append(
            EventRecord(
                "N",
                player=player,
                tile_id=called_tile if called_tile >= 0 else (tiles[0] if tiles else -1),
                junme=self.junme,
                data0=float(NAKI_TYPE_TO_CODE[info.naki_type]),
                data1=float(from_who + 1),
            )
        )
        if info.naki_type in {NakiType.CHI, NakiType.PON}:
            self.pending_player = player
            self.pending_source = "call"
        else:
            self.pending_player = -1
            self.pending_source = ""

    def process_terminal(self, tag: str, attrib: Dict[str, str]) -> None:
        event_type = "AGARI" if tag == "AGARI" else "RYUUKYOKU"
        self.event_history.append(EventRecord(event_type, player=int(attrib.get("who", -1)), junme=self.junme))
        self.pending_player = -1
        self.pending_source = ""

    def _resolve_consumed_tiles(self, player: int, candidates: Iterable[int]) -> List[int]:
        resolved: List[int] = []
        hand_counter = Counter(self.private_hands[player])
        used_exact: Counter[int] = Counter()
        for candidate in candidates:
            if hand_counter[candidate] > used_exact[candidate]:
                resolved.append(candidate)
                used_exact[candidate] += 1
                continue
            kind = tile_id_to_kind(candidate)
            replacement = next((tile for tile in self.private_hands[player] if tile_id_to_kind(tile) == kind and tile not in resolved), -1)
            if replacement >= 0:
                resolved.append(replacement)
        return resolved

    def _remove_tile_by_id_or_kind(self, player: int, tile_id: int) -> None:
        hand = self.private_hands[player]
        if tile_id in hand:
            hand.remove(tile_id)
            return
        kind = tile_id_to_kind(tile_id)
        for held in list(hand):
            if tile_id_to_kind(held) == kind:
                hand.remove(held)
                return


def row_from_observation(
    observed: ObservedState,
    label_tile_id: int,
    metadata: Dict[str, Any],
) -> Optional[DatasetRow]:
    label = tile_id_to_kind(label_tile_id)
    if label < 0:
        return None
    valid_mask = observed.valid_mask()
    if valid_mask[label] <= 0:
        return None
    return DatasetRow(
        static_features=observed.static_features(),
        sequence_features=observed.sequence_features(),
        hand_counts=observed.hand_counts(),
        aka_flags=observed.aka_flags(),
        valid_mask=valid_mask,
        label=label,
        metadata=metadata,
    )


def iter_xml_files(input_path: Path, limit_files: Optional[int] = None) -> List[Path]:
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.xml"))
    return files[:limit_files] if limit_files else files


def build_dataset_rows_from_xml(
    input_path: str | Path,
    *,
    limit_files: Optional[int] = None,
    include_call_discards: bool = True,
) -> Tuple[List[DatasetRow], ExtractionReport]:
    rows: List[DatasetRow] = []
    report = ExtractionReport()
    files = iter_xml_files(Path(input_path), limit_files)

    for xml_file in files:
        try:
            meta, rounds = parse_tenhou_xml(str(xml_file))
        except Exception as exc:  # noqa: BLE001 - extraction should continue.
            report.skipped_parse_errors += 1
            report.errors.append(f"{xml_file}: {exc}")
            continue

        report.files_processed += 1
        if meta.is_sanma:
            report.skipped_sanma += 1
            continue

        for round_data in rounds:
            report.rounds_processed += 1
            state = PrivateRoundState.from_round(round_data)
            for event_index, event in enumerate(round_data.events):
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
                            report.skipped_invalid_label += 1
                        else:
                            rows.append(row)
                            report.samples += 1
                            if state.pending_source == "call":
                                report.call_discards += 1
                            else:
                                report.draw_discards += 1
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

    return rows, report


def _parse_draw_tag(tag: str) -> Optional[Tuple[int, int]]:
    if len(tag) < 2:
        return None
    prefix = tag[0].upper()
    if prefix in DRAW_TAGS and tag[1:].isdigit():
        return DRAW_TAGS[prefix], int(tag[1:])
    return None


def _parse_discard_tag(tag: str) -> Optional[Tuple[int, int, bool]]:
    if len(tag) < 2:
        return None
    prefix = tag[0].upper()
    if prefix in DISCARD_TAGS and tag[1:].isdigit():
        return DISCARD_TAGS[prefix], int(tag[1:]), tag[0].islower()
    return None


def rows_to_npz_dict(rows: List[DatasetRow]) -> Dict[str, np.ndarray]:
    if not rows:
        return {
            "static_features": np.zeros((0, STATIC_FEATURE_DIM), dtype=np.float32),
            "sequence_features": np.zeros((0, MAX_EVENT_HISTORY, EVENT_FEATURE_DIM), dtype=np.float32),
            "hand_counts": np.zeros((0, NUM_TILE_TYPES), dtype=np.float32),
            "aka_flags": np.zeros((0, 3), dtype=np.float32),
            "valid_masks": np.zeros((0, NUM_TILE_TYPES), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "metadata": np.array([], dtype=object),
        }

    return {
        "static_features": np.stack([row.static_features for row in rows]).astype(np.float32),
        "sequence_features": np.stack([row.sequence_features for row in rows]).astype(np.float32),
        "hand_counts": np.stack([row.hand_counts for row in rows]).astype(np.float32),
        "aka_flags": np.stack([row.aka_flags for row in rows]).astype(np.float32),
        "valid_masks": np.stack([row.valid_mask for row in rows]).astype(np.float32),
        "labels": np.array([row.label for row in rows], dtype=np.int64),
        "metadata": np.array([row.metadata for row in rows], dtype=object),
    }


def validate_no_private_leakage(data: Dict[str, np.ndarray]) -> List[str]:
    errors: List[str] = []
    banned_tokens = ("opponent", "private", "hands_all", "hai1", "hai2", "hai3")
    for key in data.keys():
        lowered = key.lower()
        if any(token in lowered for token in banned_tokens):
            errors.append(f"banned private-information key found: {key}")

    labels = data.get("labels")
    valid_masks = data.get("valid_masks")
    if labels is not None and valid_masks is not None:
        for i, label in enumerate(labels):
            if valid_masks[i, int(label)] <= 0:
                errors.append(f"row {i}: label {label} is not legal in valid_mask")
                if len(errors) > 20:
                    break

    hand_counts = data.get("hand_counts")
    static_features = data.get("static_features")
    if hand_counts is not None and static_features is not None and len(hand_counts):
        static_hand = static_features[:, 13 : 13 + NUM_TILE_TYPES]
        if not np.array_equal(static_hand, hand_counts):
            errors.append("static hand block does not match actor hand_counts")
    return errors


def save_rows_npz(rows: List[DatasetRow], output_path: str | Path) -> Dict[str, np.ndarray]:
    data = rows_to_npz_dict(rows)
    errors = validate_no_private_leakage(data)
    if errors:
        raise ValueError("Leakage validation failed: " + "; ".join(errors[:5]))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    return data
