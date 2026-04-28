# tile_utils.py
"""
天鳳の牌ID管理ユーティリティ

牌ID体系（天鳳フォーマット）:
- 萬子: 0-35  (1m: 0-3, 2m: 4-7, ..., 9m: 32-35)
- 筒子: 36-71 (1p: 36-39, ..., 9p: 68-71)
- 索子: 72-107 (1s: 72-75, ..., 9s: 104-107)
- 字牌: 108-135 (東: 108-111, 南: 112-115, ..., 中: 132-135)

赤ドラ (Aka Dora):
- 赤5萬: 16 (通常5mの1枚目)
- 赤5筒: 52 (通常5pの1枚目)
- 赤5索: 88 (通常5sの1枚目)

牌種インデックス (Tile Kind): 0-33
- 萬子: 0-8 (1m-9m)
- 筒子: 9-17 (1p-9p)
- 索子: 18-26 (1s-9s)
- 字牌: 27-33 (東南西北白發中)
"""

from typing import Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
from enum import IntEnum

# =============================================================================
# 定数
# =============================================================================

# 赤ドラのタイルID
AKA_MAN_ID = 16   # 赤5萬
AKA_PIN_ID = 52   # 赤5筒
AKA_SOU_ID = 88   # 赤5索
AKA_DORA_IDS = frozenset({AKA_MAN_ID, AKA_PIN_ID, AKA_SOU_ID})

# 牌ID範囲
TILE_ID_MIN = 0
TILE_ID_MAX = 135
NUM_TILE_TYPES = 34   # 牌種数 (0-33)
NUM_TILES_TOTAL = 136 # 総牌数

# スート(色)
class Suit(IntEnum):
    MAN = 0   # 萬子
    PIN = 1   # 筒子
    SOU = 2   # 索子
    HONOR = 3 # 字牌

# 字牌のインデックス
class Honor(IntEnum):
    EAST = 27   # 東
    SOUTH = 28  # 南
    WEST = 29   # 西
    NORTH = 30  # 北
    WHITE = 31  # 白
    GREEN = 32  # 發
    RED = 33    # 中

# 牌名（表示用）
TILE_NAMES_SHORT = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",  # 0-8
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",  # 9-17
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",  # 18-26
    "東", "南", "西", "北", "白", "發", "中"              # 27-33
]

HONOR_NAMES = ["東", "南", "西", "北", "白", "發", "中"]


# =============================================================================
# データクラス
# =============================================================================

@dataclass(frozen=True, slots=True)
class TileInfo:
    """牌の詳細情報を保持するデータクラス"""
    tile_id: int       # 元の牌ID (0-135)
    kind: int          # 牌種インデックス (0-33)
    is_aka: bool       # 赤ドラかどうか
    suit: Suit         # スート
    number: int        # 数牌の場合の数字 (1-9)、字牌の場合は0
    
    def __str__(self) -> str:
        if self.is_aka:
            return f"0{['m', 'p', 's'][self.suit]}"  # 赤5は0m, 0p, 0sで表示
        return TILE_NAMES_SHORT[self.kind]
    
    def __repr__(self) -> str:
        return f"TileInfo(id={self.tile_id}, kind={self.kind}, aka={self.is_aka})"


# =============================================================================
# 基本変換関数
# =============================================================================

def is_valid_tile_id(tile_id: int) -> bool:
    """牌IDが有効かどうかを判定"""
    return isinstance(tile_id, int) and TILE_ID_MIN <= tile_id <= TILE_ID_MAX


def is_valid_tile_kind(kind: int) -> bool:
    """牌種インデックスが有効かどうかを判定"""
    return isinstance(kind, int) and 0 <= kind < NUM_TILE_TYPES


def is_aka_dora(tile_id: int) -> bool:
    """
    赤ドラかどうかを判定
    
    Args:
        tile_id: 牌ID (0-135)
    
    Returns:
        True if 赤5萬/赤5筒/赤5索
    """
    return tile_id in AKA_DORA_IDS


def tile_id_to_kind(tile_id: int) -> int:
    """
    牌ID (0-135) を牌種インデックス (0-33) に変換
    
    天鳳の牌ID体系:
    - 萬子: 0-35 (1m: 0-3, 2m: 4-7, ..., 5m: 16-19[16が赤], ..., 9m: 32-35)
    - 筒子: 36-71 (1p: 36-39, ..., 5p: 52-55[52が赤], ..., 9p: 68-71)
    - 索子: 72-107 (1s: 72-75, ..., 5s: 88-91[88が赤], ..., 9s: 104-107)
    - 字牌: 108-135 (東: 108-111, ..., 中: 132-135)
    
    赤ドラは対応する5の牌種に変換される:
    - 赤5萬(16) -> 4 (5萬のkind)
    - 赤5筒(52) -> 13 (5筒のkind)
    - 赤5索(88) -> 22 (5索のkind)
    
    Args:
        tile_id: 牌ID (0-135)
    
    Returns:
        牌種インデックス (0-33)、無効な場合は -1
    """
    if not is_valid_tile_id(tile_id):
        return -1
    
    # 数牌 (0-107)
    if tile_id < 108:
        return tile_id // 4
    # 字牌 (108-135)
    else:
        return 27 + (tile_id - 108) // 4


# 互換性のためのエイリアス
tile_id_to_index = tile_id_to_kind


def tile_kind_to_id(kind: int, prefer_aka: bool = False, offset: int = 0) -> int:
    """
    牌種インデックス (0-33) を牌ID (0-135) に変換
    
    Args:
        kind: 牌種インデックス (0-33)
        prefer_aka: Trueの場合、5の牌は赤ドラIDを返す
        offset: 同一牌種内のオフセット (0-3)
    
    Returns:
        牌ID (0-135)、無効な場合は -1
    """
    if not is_valid_tile_kind(kind):
        return -1

    offset = max(0, min(3, offset))  # 0-3にクランプ
    
    # 5の牌で赤ドラを優先する場合
    if prefer_aka and offset == 0:
        if kind == 4:   # 5萬
            return AKA_MAN_ID
        elif kind == 13:  # 5筒
            return AKA_PIN_ID
        elif kind == 22:  # 5索
            return AKA_SOU_ID
    
    # 数牌 (kind 0-26)
    if kind < 27:
        return kind * 4 + offset
    # 字牌 (kind 27-33)
    else:
        return 108 + (kind - 27) * 4 + offset


# 互換性のためのエイリアス
tile_index_to_id = tile_kind_to_id


def tile_id_to_string(tile_id: int, show_aka: bool = True) -> str:
    """
    牌IDを文字列表現に変換
    
    Args:
        tile_id: 牌ID (0-135)
        show_aka: 赤ドラを "0m", "0p", "0s" で表示する場合True
    
    Returns:
        牌の文字列表現、無効な場合は "?"
    """
    if not is_valid_tile_id(tile_id):
        return "?"
    
    # 赤ドラの表示
    if show_aka:
        if tile_id == AKA_MAN_ID:
            return "0m"
        elif tile_id == AKA_PIN_ID:
            return "0p"
        elif tile_id == AKA_SOU_ID:
            return "0s"
    
    kind = tile_id_to_kind(tile_id)
    if kind == -1:
        return "?"

    return TILE_NAMES_SHORT[kind]


def tile_kind_to_string(kind: int) -> str:
    """
    牌種インデックスを文字列表現に変換
    
    Args:
        kind: 牌種インデックス (0-33)
    
    Returns:
        牌の文字列表現、無効な場合は "?"
    """
    if not is_valid_tile_kind(kind):
        return "?"
    return TILE_NAMES_SHORT[kind]


def get_tile_info(tile_id: int) -> Optional[TileInfo]:
    """
    牌IDから詳細情報を取得
    
    Args:
        tile_id: 牌ID (0-135)
    
    Returns:
        TileInfo オブジェクト、無効な場合は None
    """
    if not is_valid_tile_id(tile_id):
        return None
    
    is_aka = is_aka_dora(tile_id)
    kind = tile_id_to_kind(tile_id)
    
    # スートと数字を計算
    if kind < 27:  # 数牌
        suit = Suit(kind // 9)
        number = (kind % 9) + 1
    else:  # 字牌
        suit = Suit.HONOR
        number = 0
    
    return TileInfo(
        tile_id=tile_id,
        kind=kind,
        is_aka=is_aka,
        suit=suit,
        number=number
    )


# =============================================================================
# 手牌操作ユーティリティ
# =============================================================================

def hand_to_kinds(hand: List[int]) -> List[int]:
    """
    手牌の牌IDリストを牌種インデックスリストに変換
    
    Args:
        hand: 牌IDのリスト
    
    Returns:
        牌種インデックスのリスト（無効なIDは除外）
    """
    return [tile_id_to_kind(t) for t in hand if tile_id_to_kind(t) != -1]


def hand_to_string(hand: List[int], sort: bool = True) -> str:
    """
    手牌を文字列表現に変換
    
    Args:
        hand: 牌IDのリスト
        sort: ソートするかどうか
    
    Returns:
        手牌の文字列表現
    """
    if sort:
        sorted_hand = sorted(hand, key=lambda t: (tile_id_to_kind(t), t))
    else:
        sorted_hand = hand
    return " ".join(tile_id_to_string(t) for t in sorted_hand)


def count_tiles_by_kind(hand: List[int]) -> dict:
    """
    手牌の牌種ごとの枚数をカウント
    
    Args:
        hand: 牌IDのリスト
    
    Returns:
        {kind: count} の辞書
    """
    counts = {}
    for tile_id in hand:
        kind = tile_id_to_kind(tile_id)
        if kind != -1:
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def count_aka_dora(hand: List[int]) -> int:
    """
    手牌の赤ドラ枚数をカウント
    
    Args:
        hand: 牌IDのリスト
    
    Returns:
        赤ドラの枚数
    """
    return sum(1 for t in hand if is_aka_dora(t))


def find_tiles_by_kind(hand: List[int], kind: int) -> List[int]:
    """
    手牌から特定の牌種の牌IDを全て取得
    
    Args:
        hand: 牌IDのリスト
        kind: 検索する牌種インデックス
    
    Returns:
        該当する牌IDのリスト
    """
    return [t for t in hand if tile_id_to_kind(t) == kind]


def is_tile_in_hand(hand: List[int], tile_id: int) -> bool:
    """
    指定した牌IDが手牌に含まれるか確認
    
    Args:
        hand: 牌IDのリスト
        tile_id: 確認する牌ID
    
    Returns:
        含まれる場合 True
    """
    return tile_id in hand


def is_kind_in_hand(hand: List[int], kind: int) -> bool:
    """
    指定した牌種が手牌に含まれるか確認
    
    Args:
        hand: 牌IDのリスト
        kind: 確認する牌種インデックス
    
    Returns:
        含まれる場合 True
    """
    return any(tile_id_to_kind(t) == kind for t in hand)


# =============================================================================
# ドラ計算
# =============================================================================

def dora_indicator_to_dora(indicator_id: int) -> int:
    """
    ドラ表示牌からドラ牌の牌種を計算
    
    Args:
        indicator_id: ドラ表示牌のID
    
    Returns:
        ドラ牌の牌種インデックス、無効な場合は -1
    """
    indicator_kind = tile_id_to_kind(indicator_id)
    if indicator_kind == -1:
        return -1
    
    # 数牌: 次の数字、9の次は1
    if indicator_kind <= 26:
        suit_base = (indicator_kind // 9) * 9  # スートの開始インデックス
        number_in_suit = indicator_kind % 9    # スート内の位置 (0-8)
        return suit_base + (number_in_suit + 1) % 9
    
    # 風牌 (東南西北): 次の風、北の次は東
    elif 27 <= indicator_kind <= 30:
        return 27 + (indicator_kind - 27 + 1) % 4
    
    # 三元牌 (白發中): 次の三元牌、中の次は白
    elif 31 <= indicator_kind <= 33:
        return 31 + (indicator_kind - 31 + 1) % 3
    
    return -1


# =============================================================================
# 検証・デバッグ用
# =============================================================================

def validate_hand(hand: List[int], expected_count: Optional[int] = None) -> Tuple[bool, List[str]]:
    """
    手牌の整合性を検証
    
    Args:
        hand: 牌IDのリスト
        expected_count: 期待される枚数（Noneの場合はチェックしない）
    
    Returns:
        (is_valid, errors) のタプル
    """
    errors = []
    
    # 枚数チェック
    if expected_count is not None and len(hand) != expected_count:
        errors.append(f"手牌枚数が不正: {len(hand)} (期待値: {expected_count})")
    
    # 牌IDの範囲チェック
    for tile_id in hand:
        if not is_valid_tile_id(tile_id):
            errors.append(f"無効な牌ID: {tile_id}")
    
    # 同一牌の枚数チェック（各牌は最大4枚）
    kind_counts = count_tiles_by_kind(hand)
    for kind, count in kind_counts.items():
        if count > 4:
            errors.append(f"牌種 {tile_kind_to_string(kind)} が5枚以上: {count}枚")
    
    return len(errors) == 0, errors


def debug_print_hand(hand: List[int], label: str = "手牌") -> None:
    """
    手牌をデバッグ用に表示
    
    Args:
        hand: 牌IDのリスト
        label: 表示ラベル
    """
    sorted_hand = sorted(hand, key=lambda t: (tile_id_to_kind(t), t))
    
    print(f"=== {label} ({len(hand)}枚) ===")
    print(f"  表示: {hand_to_string(sorted_hand, sort=False)}")
    print(f"  IDs:  {sorted_hand}")
    
    # 赤ドラの有無
    aka_count = count_aka_dora(hand)
    if aka_count > 0:
        aka_tiles = [tile_id_to_string(t) for t in hand if is_aka_dora(t)]
        print(f"  赤牌: {' '.join(aka_tiles)} ({aka_count}枚)")
    
    # 牌種別カウント
    kind_counts = count_tiles_by_kind(hand)
    kind_str = " ".join(f"{tile_kind_to_string(k)}:{c}" for k, c in sorted(kind_counts.items()))
    print(f"  種類: {kind_str}")


# =============================================================================
# 特徴量生成用
# =============================================================================

def hand_to_34_array(hand: List[int]) -> List[int]:
    """
    手牌を34種の牌種カウント配列に変換
    
    Args:
        hand: 牌IDのリスト
    
    Returns:
        長さ34の配列（各要素は0-4の枚数）
    """
    counts = [0] * NUM_TILE_TYPES
    for tile_id in hand:
        kind = tile_id_to_kind(tile_id)
        if kind != -1:
            counts[kind] += 1
    return counts


def hand_to_34_with_aka(hand: List[int]) -> Tuple[List[int], List[int]]:
    """
    手牌を34種の牌種カウント配列と赤ドラ配列に変換
    
    Args:
        hand: 牌IDのリスト
    
    Returns:
        (counts, aka_flags) のタプル
        - counts: 長さ34の配列（各要素は0-4の枚数）
        - aka_flags: 長さ3の配列（[赤5m有無, 赤5p有無, 赤5s有無]）
    """
    counts = hand_to_34_array(hand)
    
    # 赤ドラフラグ
    aka_flags = [
        1 if AKA_MAN_ID in hand else 0,
        1 if AKA_PIN_ID in hand else 0,
        1 if AKA_SOU_ID in hand else 0,
    ]
    
    return counts, aka_flags


# =============================================================================
# テスト・検証
# =============================================================================

def run_self_tests() -> bool:
    """
    自己検査を実行
    
    Returns:
        全テスト通過で True
    """
    all_passed = True
    
    # テスト1: 赤ドラ判定
    assert is_aka_dora(AKA_MAN_ID), "赤5萬の判定失敗"
    assert is_aka_dora(AKA_PIN_ID), "赤5筒の判定失敗"
    assert is_aka_dora(AKA_SOU_ID), "赤5索の判定失敗"
    assert not is_aka_dora(0), "通常牌の赤ドラ誤判定"
    assert not is_aka_dora(17), "通常5萬の赤ドラ誤判定"
    
    # テスト2: 牌ID→牌種変換
    # 天鳳の牌ID: 1m=0-3, 2m=4-7, 3m=8-11, 4m=12-15, 5m=16-19(赤16), 6m=20-23, ...
    assert tile_id_to_kind(0) == 0, f"1萬の変換失敗: {tile_id_to_kind(0)}"
    assert tile_id_to_kind(4) == 1, f"2萬の変換失敗: {tile_id_to_kind(4)}"
    assert tile_id_to_kind(AKA_MAN_ID) == 4, f"赤5萬の変換失敗: {tile_id_to_kind(AKA_MAN_ID)}"
    assert tile_id_to_kind(17) == 4, f"通常5萬の変換失敗: {tile_id_to_kind(17)}"
    assert tile_id_to_kind(20) == 5, f"6萬の変換失敗: {tile_id_to_kind(20)}"
    assert tile_id_to_kind(108) == 27, f"東の変換失敗: {tile_id_to_kind(108)}"
    assert tile_id_to_kind(132) == 33, f"中の変換失敗: {tile_id_to_kind(132)}"
    assert tile_id_to_kind(-1) == -1, "無効IDの変換失敗"
    assert tile_id_to_kind(136) == -1, "範囲外IDの変換失敗"
    
    # テスト3: 牌種→牌ID変換
    assert tile_kind_to_id(0) == 0, "1萬の逆変換失敗"
    assert tile_kind_to_id(4, prefer_aka=True) == AKA_MAN_ID, "赤5萬優先の逆変換失敗"
    assert tile_kind_to_id(4, prefer_aka=False) == 16, "通常5萬の逆変換失敗（offset=0、赤IDと同じ）"
    assert tile_kind_to_id(27) == 108, "東の逆変換失敗"
    
    # テスト4: 文字列変換
    assert tile_id_to_string(0) == "1m", "1萬の文字列変換失敗"
    assert tile_id_to_string(AKA_MAN_ID) == "0m", "赤5萬の文字列変換失敗"
    assert tile_id_to_string(17) == "5m", "通常5萬の文字列変換失敗"
    assert tile_id_to_string(108) == "東", "東の文字列変換失敗"
    
    # テスト5: ドラ計算
    # 1萬表示(kind=0) → 2萬ドラ(kind=1)
    assert dora_indicator_to_dora(0) == 1, f"1萬表示→2萬ドラ失敗: {dora_indicator_to_dora(0)}"
    # 9萬表示(kind=8) → 1萬ドラ(kind=0)
    assert dora_indicator_to_dora(32) == 0, f"9萬表示→1萬ドラ失敗: {dora_indicator_to_dora(32)}"
    # 東表示(kind=27) → 南ドラ(kind=28)
    assert dora_indicator_to_dora(108) == 28, f"東表示→南ドラ失敗: {dora_indicator_to_dora(108)}"
    # 北表示(kind=30) → 東ドラ(kind=27)
    assert dora_indicator_to_dora(120) == 27, f"北表示→東ドラ失敗: {dora_indicator_to_dora(120)}"
    # 中表示(kind=33) → 白ドラ(kind=31)
    assert dora_indicator_to_dora(132) == 31, f"中表示→白ドラ失敗: {dora_indicator_to_dora(132)}"
    
    # テスト6: 手牌検証
    # 1m, 2m, 3m, 4m, 赤5m, 1p, 2p, 3p, 4p, 赤5p, 1s, 2s, 3s
    valid_hand = [0, 4, 8, 12, AKA_MAN_ID, 36, 40, 44, 48, AKA_PIN_ID, 72, 76, 80]  # 13枚
    is_valid, errors = validate_hand(valid_hand, expected_count=13)
    assert is_valid, f"有効手牌の検証失敗: {errors}"
    
    invalid_hand = [0, 0, 0, 0, 0]  # 同じIDが5枚（ありえない）
    is_valid, errors = validate_hand(invalid_hand)
    assert not is_valid, "無効手牌（5枚以上）の検証が通過"
    
    print("[tile_utils] 全テスト通過")
    return all_passed


if __name__ == "__main__":
    run_self_tests()
