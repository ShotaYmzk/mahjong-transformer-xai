# feature_utils.py
"""
特徴量生成のユーティリティ

麻雀の状態から学習用特徴量を生成する関数群。
training時やprediction時に共通で使用。
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

# tile_utilsのインポート
_utils_dir = Path(__file__).parent
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))

from tile_utils import (
    tile_id_to_kind, is_aka_dora, is_valid_tile_id,
    NUM_TILE_TYPES, AKA_MAN_ID, AKA_PIN_ID, AKA_SOU_ID,
    dora_indicator_to_dora
)


# =============================================================================
# 定数
# =============================================================================

NUM_PLAYERS = 4
MAX_RIVER_LENGTH = 30   # 河の最大長（特徴量用）


# =============================================================================
# 手牌特徴量
# =============================================================================

def hand_to_one_hot(hand: List[int]) -> np.ndarray:
    """
    手牌をone-hot形式に変換（34×4の形式）
    
    Args:
        hand: 牌IDのリスト
    
    Returns:
        shape (34, 4) の配列。各牌種について4枚分のone-hot
    """
    features = np.zeros((NUM_TILE_TYPES, 4), dtype=np.float32)
    
    kind_count = {}
    for tile_id in hand:
        kind = tile_id_to_kind(tile_id)
        if kind != -1:
            count = kind_count.get(kind, 0)
            if count < 4:
                features[kind, count] = 1.0
                kind_count[kind] = count + 1
    
    return features


def hand_to_count(hand: List[int]) -> np.ndarray:
    """
    手牌を枚数カウント形式に変換
    
    Args:
        hand: 牌IDのリスト
    
    Returns:
        shape (34,) の配列。各牌種の枚数（0-4）
    """
    counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    
    for tile_id in hand:
        kind = tile_id_to_kind(tile_id)
        if 0 <= kind < NUM_TILE_TYPES:
            counts[kind] += 1
    
    return counts


def hand_to_count_with_aka(hand: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    手牌を枚数カウントと赤ドラフラグに変換
    
    Args:
        hand: 牌IDのリスト
    
    Returns:
        (counts, aka_flags):
            counts: shape (34,) の配列
            aka_flags: shape (3,) の配列 [赤5m, 赤5p, 赤5s]
    """
    counts = hand_to_count(hand)
    
    aka_flags = np.array([
        1.0 if AKA_MAN_ID in hand else 0.0,
        1.0 if AKA_PIN_ID in hand else 0.0,
        1.0 if AKA_SOU_ID in hand else 0.0,
    ], dtype=np.float32)
    
    return counts, aka_flags


def create_valid_action_mask(hand: List[int], is_riichi: bool = False) -> np.ndarray:
    """
    合法手マスクを生成
    
    Args:
        hand: 手牌（ツモ後14枚の状態）
        is_riichi: リーチ中かどうか
    
    Returns:
        shape (34,) のマスク配列（打牌可能な牌種が1）
    """
    mask = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    
    if is_riichi and hand:
        # リーチ後はツモ切りのみ
        last_tile = hand[-1]
        kind = tile_id_to_kind(last_tile)
        if 0 <= kind < NUM_TILE_TYPES:
            mask[kind] = 1.0
    else:
        # 通常時は手牌にある牌種が全て合法
        for tile_id in hand:
            kind = tile_id_to_kind(tile_id)
            if 0 <= kind < NUM_TILE_TYPES:
                mask[kind] = 1.0
    
    return mask


# =============================================================================
# 河・捨て牌特徴量
# =============================================================================

def river_to_sequence(
    discards: List[Tuple[int, bool]], 
    max_length: int = MAX_RIVER_LENGTH
) -> np.ndarray:
    """
    河（捨て牌リスト）を系列特徴量に変換
    
    Args:
        discards: [(tile_id, tsumogiri), ...] のリスト
        max_length: 最大長（パディング用）
    
    Returns:
        shape (max_length, 2) の配列 [tile_kind+1, tsumogiri_flag]
    """
    features = np.zeros((max_length, 2), dtype=np.float32)
    
    for i, (tile_id, tsumogiri) in enumerate(discards[:max_length]):
        kind = tile_id_to_kind(tile_id)
        features[i, 0] = kind + 1 if kind != -1 else 0  # 0はパディング
        features[i, 1] = 1.0 if tsumogiri else 0.0
    
    return features


def river_to_count(discards: List[Tuple[int, bool]]) -> np.ndarray:
    """
    河の牌種カウントを取得
    
    Args:
        discards: [(tile_id, tsumogiri), ...] のリスト
    
    Returns:
        shape (34,) の配列
    """
    counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    
    for tile_id, _ in discards:
        kind = tile_id_to_kind(tile_id)
        if 0 <= kind < NUM_TILE_TYPES:
            counts[kind] += 1
    
    return counts


def visible_tiles_count(
    all_discards: List[List[Tuple[int, bool]]],
    all_melds: List[List[Dict]],
    dora_indicators: List[int] = None
) -> np.ndarray:
    """
    全員に見えている牌のカウントを取得
    
    Args:
        all_discards: 全プレイヤーの河
        all_melds: 全プレイヤーの副露
        dora_indicators: ドラ表示牌
    
    Returns:
        shape (34,) の配列（各牌種の見えている枚数）
    """
    counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    
    # 河
    for discards in all_discards:
        for tile_id, _ in discards:
            kind = tile_id_to_kind(tile_id)
            if 0 <= kind < NUM_TILE_TYPES:
                counts[kind] += 1
    
    # 副露
    for melds in all_melds:
        for meld in melds:
            for tile_id in meld.get('tiles', []):
                kind = tile_id_to_kind(tile_id)
                if 0 <= kind < NUM_TILE_TYPES:
                    counts[kind] += 1
    
    # ドラ表示牌
    if dora_indicators:
        for tile_id in dora_indicators:
            kind = tile_id_to_kind(tile_id)
            if 0 <= kind < NUM_TILE_TYPES:
                counts[kind] += 1
    
    return counts


def remaining_tiles_count(
    hand: List[int],
    visible_counts: np.ndarray
) -> np.ndarray:
    """
    残り牌数（見えていない牌）を計算
    
    Args:
        hand: 自分の手牌
        visible_counts: 見えている牌のカウント
    
    Returns:
        shape (34,) の配列（各牌種の残り枚数）
    """
    # 各牌種は4枚ずつ
    remaining = np.full(NUM_TILE_TYPES, 4.0, dtype=np.float32)
    
    # 見えている牌を引く
    remaining -= visible_counts
    
    # 自分の手牌を引く
    for tile_id in hand:
        kind = tile_id_to_kind(tile_id)
        if 0 <= kind < NUM_TILE_TYPES:
            remaining[kind] -= 1
    
    # 0未満にならないようにクリップ
    remaining = np.maximum(remaining, 0.0)
    
    return remaining


# =============================================================================
# ドラ関連
# =============================================================================

def dora_count_in_hand(hand: List[int], dora_indicators: List[int]) -> int:
    """
    手牌のドラ枚数を計算
    
    Args:
        hand: 手牌
        dora_indicators: ドラ表示牌のリスト
    
    Returns:
        ドラ枚数
    """
    dora_kinds = set()
    for ind_id in dora_indicators:
        dora_kind = dora_indicator_to_dora(ind_id)
        if dora_kind != -1:
            dora_kinds.add(dora_kind)
    
    count = 0
    for tile_id in hand:
        kind = tile_id_to_kind(tile_id)
        if kind in dora_kinds:
            count += 1
    
    return count


def aka_dora_count_in_hand(hand: List[int]) -> int:
    """
    手牌の赤ドラ枚数を計算
    
    Args:
        hand: 手牌
    
    Returns:
        赤ドラ枚数
    """
    return sum(1 for t in hand if is_aka_dora(t))


def dora_indicator_features(dora_indicators: List[int]) -> np.ndarray:
    """
    ドラ表示牌の特徴量
    
    Args:
        dora_indicators: ドラ表示牌のリスト
    
    Returns:
        shape (34,) の配列（各牌種がドラ表示牌かどうか）
    """
    features = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    
    for ind_id in dora_indicators:
        kind = tile_id_to_kind(ind_id)
        if 0 <= kind < NUM_TILE_TYPES:
            features[kind] += 1.0
    
    return features


def dora_tiles_features(dora_indicators: List[int]) -> np.ndarray:
    """
    ドラ牌の特徴量（表示牌からドラを計算）
    
    Args:
        dora_indicators: ドラ表示牌のリスト
    
    Returns:
        shape (34,) の配列（各牌種がドラかどうか）
    """
    features = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    
    for ind_id in dora_indicators:
        dora_kind = dora_indicator_to_dora(ind_id)
        if 0 <= dora_kind < NUM_TILE_TYPES:
            features[dora_kind] += 1.0
    
    return features


# =============================================================================
# プレイヤー状態特徴量
# =============================================================================

def player_position_features(
    my_id: int, 
    num_players: int = NUM_PLAYERS
) -> np.ndarray:
    """
    自分から見た各プレイヤーの相対位置
    
    Args:
        my_id: 自分のプレイヤーID
        num_players: プレイヤー数
    
    Returns:
        shape (num_players,) の配列（自分が1、他は0）
    """
    features = np.zeros(num_players, dtype=np.float32)
    features[0] = 1.0  # 自分
    return features


def riichi_status_features(
    riichi_status: List[int],
    my_id: int
) -> np.ndarray:
    """
    各プレイヤーのリーチ状態
    
    Args:
        riichi_status: [0,1,2,...] のリスト（0:なし, 1:宣言, 2:確定）
        my_id: 自分のプレイヤーID
    
    Returns:
        shape (NUM_PLAYERS,) の配列（リーチ中が1）
    """
    features = np.zeros(NUM_PLAYERS, dtype=np.float32)
    
    for i in range(NUM_PLAYERS):
        # 自分から見た相対位置
        rel_pos = (i - my_id + NUM_PLAYERS) % NUM_PLAYERS
        features[rel_pos] = 1.0 if riichi_status[i] == 2 else 0.0
    
    return features


# =============================================================================
# 統合特徴量生成
# =============================================================================

def create_full_features(
    hand: List[int],
    discards: List[List[Tuple[int, bool]]],
    melds: List[List[Dict]],
    dora_indicators: List[int],
    riichi_status: List[int],
    my_id: int,
    junme: float,
    is_dealer: bool,
    round_wind: int = 0,
    honba: int = 0,
    kyotaku: int = 0,
    wall_count: int = 70,
) -> Dict[str, np.ndarray]:
    """
    全ての特徴量を生成
    
    Args:
        hand: 自分の手牌
        discards: 全プレイヤーの河
        melds: 全プレイヤーの副露
        dora_indicators: ドラ表示牌
        riichi_status: 各プレイヤーのリーチ状態
        my_id: 自分のプレイヤーID
        junme: 巡目
        is_dealer: 自分が親かどうか
        round_wind: 場風
        honba: 本場
        kyotaku: 供託
        wall_count: 残り山牌数
    
    Returns:
        特徴量の辞書
    """
    # 手牌
    hand_count, aka_flags = hand_to_count_with_aka(hand)
    
    # 自分の河
    my_river = river_to_count(discards[my_id])
    
    # 見えている牌
    visible = visible_tiles_count(discards, melds, dora_indicators)
    
    # 残り牌
    remaining = remaining_tiles_count(hand, visible)
    
    # ドラ
    dora_features = dora_tiles_features(dora_indicators)
    
    # リーチ状態
    riichi_features = riichi_status_features(riichi_status, my_id)
    
    # 合法手マスク
    is_riichi = riichi_status[my_id] == 2
    valid_mask = create_valid_action_mask(hand, is_riichi)
    
    # コンテキスト特徴量
    context = np.array([
        round_wind,
        honba,
        kyotaku,
        junme,
        wall_count,
        float(is_dealer),
        len(melds[my_id]),  # 副露数
        riichi_status[my_id],
    ], dtype=np.float32)
    
    return {
        'hand_count': hand_count,          # (34,)
        'aka_flags': aka_flags,            # (3,)
        'my_river': my_river,              # (34,)
        'visible': visible,                # (34,)
        'remaining': remaining,            # (34,)
        'dora': dora_features,             # (34,)
        'riichi': riichi_features,         # (4,)
        'valid_mask': valid_mask,          # (34,)
        'context': context,                # (8,)
    }


# =============================================================================
# テスト
# =============================================================================

def run_self_tests() -> bool:
    """自己検査"""
    print("[feature_utils] 自己検査開始...")
    
    # テスト手牌
    hand = [0, 4, 8, 12, 16, 36, 40, 44, 48, 52, 72, 76, 80]
    
    # hand_to_count テスト
    counts = hand_to_count(hand)
    assert counts.shape == (34,)
    assert counts[0] == 1  # 1m
    assert counts[4] == 1  # 5m (赤5mから)
    
    # hand_to_count_with_aka テスト
    counts, aka = hand_to_count_with_aka(hand)
    assert aka[0] == 1  # 赤5m
    assert aka[1] == 1  # 赤5p
    assert aka[2] == 0  # 赤5sなし
    
    # 合法手マスク テスト
    hand_14 = hand + [84]  # 14枚
    mask = create_valid_action_mask(hand_14)
    assert mask.sum() > 0
    
    # river_to_count テスト
    discards = [(0, False), (4, True)]
    river_count = river_to_count(discards)
    assert river_count[0] == 1
    assert river_count[1] == 1
    
    print("[feature_utils] 全テスト通過")
    return True


if __name__ == "__main__":
    run_self_tests()


