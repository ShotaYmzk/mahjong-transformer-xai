# naki_utils.py
"""
天鳳の副露（鳴き）デコードユーティリティ

天鳳XMLログの <N> タグの m 属性をデコードし、
鳴きの種類・構成牌・鳴き元を解析する。

参考: https://gimite.net/pukiwiki/index.php?Tenhou%20Log%20Format
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# インポート設定
# utils ディレクトリから tile_utils をインポート
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from .tile_utils import (
        tile_id_to_kind, tile_id_to_string, is_aka_dora,
        is_valid_tile_id, tile_kind_to_id
    )
    TILE_UTILS_AVAILABLE = True
except ImportError as e:
    try:
        from tile_utils import (
            tile_id_to_kind, tile_id_to_string, is_aka_dora,
            is_valid_tile_id, tile_kind_to_id
        )
        TILE_UTILS_AVAILABLE = True
    except ImportError:
        raise ImportError(
            "Cannot import tile_utils. Import this module as `utils.naki_utils` "
            "or run from the repository root."
        ) from e


# =============================================================================
# 定数・列挙型
# =============================================================================

class NakiType(Enum):
    """鳴きの種類"""
    CHI = "チー"
    PON = "ポン"
    DAIMINKAN = "大明槓"
    KAKAN = "加槓"
    ANKAN = "暗槓"
    UNKNOWN = "不明"


# 鳴きタイプから整数コードへのマッピング（特徴量用）
NAKI_TYPE_CODES = {
    NakiType.CHI: 0,
    NakiType.PON: 1,
    NakiType.DAIMINKAN: 2,
    NakiType.KAKAN: 3,
    NakiType.ANKAN: 4,
    NakiType.UNKNOWN: -1,
}

# 文字列からNakiTypeへのマッピング（互換性用）
NAKI_TYPE_FROM_STRING = {
    "チー": NakiType.CHI,
    "ポン": NakiType.PON,
    "大明槓": NakiType.DAIMINKAN,
    "加槓": NakiType.KAKAN,
    "暗槓": NakiType.ANKAN,
    "不明": NakiType.UNKNOWN,
}


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class NakiInfo:
    """
    鳴き情報を保持するデータクラス
    
    Attributes:
        naki_type: 鳴きの種類
        tiles: 構成牌のIDリスト（鳴いた牌を含む）
        from_who_relative: 鳴き元の相対位置 (0:下家, 1:対面, 2:上家), -1:自分
        consumed: 手牌から消費された牌IDのリスト
        called_tile_id: 鳴かれた牌のID（チー・ポン・大明槓の場合）
        raw_m_value: 元のmビットフィールド値
        decode_error: デコードエラーメッセージ（あれば）
    """
    naki_type: NakiType = NakiType.UNKNOWN
    tiles: List[int] = field(default_factory=list)
    from_who_relative: int = -1
    consumed: List[int] = field(default_factory=list)
    called_tile_id: int = -1
    raw_m_value: int = 0
    decode_error: Optional[str] = None
    
    @property
    def type_string(self) -> str:
        """互換性のためのプロパティ"""
        return self.naki_type.value
    
    @property
    def type_code(self) -> int:
        """鳴きタイプの整数コード"""
        return NAKI_TYPE_CODES.get(self.naki_type, -1)
    
    def is_valid(self) -> bool:
        """有効な鳴き情報かどうか"""
        return self.naki_type != NakiType.UNKNOWN and len(self.tiles) > 0
    
    def to_dict(self) -> Dict:
        """辞書形式に変換（互換性用）"""
        return {
            "type": self.naki_type.value,
            "tiles": self.tiles,
            "from_who_relative": self.from_who_relative,
            "consumed": self.consumed,
            "raw_value": self.raw_m_value,
        }
    
    def __str__(self) -> str:
        tiles_str = " ".join(tile_id_to_string(t) for t in self.tiles)
        return f"{self.naki_type.value}: [{tiles_str}]"


# =============================================================================
# デコード関数
# =============================================================================

def decode_naki(m: int) -> NakiInfo:
    """
    天鳳の副露面子のビットフィールド(m)をデコードする
    
    Args:
        m: <N>タグの m 属性の値
    
    Returns:
        NakiInfo: デコードされた鳴き情報
    
    Note:
        - チー・ポン・大明槓では called_tile_id は特定できない場合がある
        - 呼び出し元で直前の捨て牌から判断する必要がある
    """
    result = NakiInfo(raw_m_value=m)
    
    try:
        # from_who_relative: 下位2ビット
        from_who_relative = m & 3
        result.from_who_relative = from_who_relative
        
        # --- チー (ビット2が立っている) ---
        if m & (1 << 2):
            result.naki_type = NakiType.CHI
            result = _decode_chi(m, result)
        
        # --- ポン (ビット3が立っている) ---
        elif m & (1 << 3):
            result.naki_type = NakiType.PON
            result = _decode_pon(m, result)
        
        # --- 加槓 (ビット4が立っている) ---
        elif m & (1 << 4):
            result.naki_type = NakiType.KAKAN
            result.from_who_relative = -1  # 自分自身
            result = _decode_kakan(m, result)
        
        # --- 大明槓 or 暗槓 (ビット2,3,4が全て0) ---
        else:
            result = _decode_kan(m, result, from_who_relative)
        
        return result
    
    except Exception as e:
        result.naki_type = NakiType.UNKNOWN
        result.decode_error = f"デコードエラー: {str(e)}"
        return result


def _decode_chi(m: int, result: NakiInfo) -> NakiInfo:
    """チーのデコード"""
    t = m >> 10
    r = t % 3  # 鳴かれた牌の位置（順子内）
    t //= 3
    
    # ベースインデックスの計算
    # t: 0-6 → 1m-7m, 7-13 → 1p-7p, 14-20 → 1s-7s
    if 0 <= t <= 6:
        base_kind = t  # 1m-7m (kind 0-6)
    elif 7 <= t <= 13:
        base_kind = (t - 7) + 9  # 1p-7p (kind 9-15)
    elif 14 <= t <= 20:
        base_kind = (t - 14) + 18  # 1s-7s (kind 18-24)
    else:
        result.decode_error = f"チー: 無効なt値 {t}"
        result.naki_type = NakiType.UNKNOWN
        return result
    
    # 各牌のオフセット（同一種内のどの牌か）
    offsets = [(m >> 3) & 3, (m >> 5) & 3, (m >> 7) & 3]
    
    tiles = []
    consumed = []
    for i in range(3):
        tile_kind = base_kind + i
        tile_id = tile_kind * 4 + offsets[i]
        
        # 赤ドラの補正（offset 0 で 5 の牌の場合）
        if offsets[i] == 0 and tile_kind in {4, 13, 22}:  # 5m, 5p, 5s
            tile_id = {4: 16, 13: 52, 22: 88}[tile_kind]  # 赤ドラID
        
        if not is_valid_tile_id(tile_id):
            result.decode_error = f"チー: 無効な牌ID {tile_id}"
            result.naki_type = NakiType.UNKNOWN
            return result
        
        tiles.append(tile_id)
        if i != r:  # 鳴かれた牌以外が手牌から消費
            consumed.append(tile_id)
    
    result.tiles = sorted(tiles)
    result.consumed = sorted(consumed)
    return result


def _decode_pon(m: int, result: NakiInfo) -> NakiInfo:
    """ポンのデコード"""
    t = m >> 9
    t //= 3
    
    if not (0 <= t <= 33):
        result.decode_error = f"ポン: 無効な牌種 {t}"
        result.naki_type = NakiType.UNKNOWN
        return result
    
    base_id = t * 4
    unused_offset = (m >> 5) & 3  # 使われなかった牌のオフセット
    
    # 4枚中3枚を使用
    tiles = []
    for i in range(4):
        if i != unused_offset:
            tile_id = base_id + i
            # 赤ドラIDへの変換
            if i == 0 and t in {4, 13, 22}:
                tile_id = {4: 16, 13: 52, 22: 88}[t]
            tiles.append(tile_id)
    
    if len(tiles) != 3:
        result.decode_error = f"ポン: 牌選択エラー (unused={unused_offset})"
        result.naki_type = NakiType.UNKNOWN
        return result
    
    result.tiles = sorted(tiles)
    # consumed は GameState で特定する（鳴かれた牌を除外するため）
    result.consumed = []
    return result


def _decode_kakan(m: int, result: NakiInfo) -> NakiInfo:
    """加槓のデコード"""
    t = m >> 9
    t //= 3
    
    if not (0 <= t <= 33):
        result.decode_error = f"加槓: 無効な牌種 {t}"
        result.naki_type = NakiType.UNKNOWN
        return result
    
    base_id = t * 4
    added_offset = (m >> 5) & 3  # 追加された牌のオフセット
    added_tile_id = base_id + added_offset
    
    # 赤ドラIDへの変換
    if added_offset == 0 and t in {4, 13, 22}:
        added_tile_id = {4: 16, 13: 52, 22: 88}[t]
    
    # 結果には4枚全てを含める
    all_tiles = []
    for i in range(4):
        tile_id = base_id + i
        if i == 0 and t in {4, 13, 22}:
            tile_id = {4: 16, 13: 52, 22: 88}[t]
        all_tiles.append(tile_id)
    
    result.tiles = sorted(all_tiles)
    result.consumed = [added_tile_id]  # 手牌から消費されたのは追加牌のみ
    result.called_tile_id = added_tile_id
    return result


def _decode_kan(m: int, result: NakiInfo, from_who_relative: int) -> NakiInfo:
    """大明槓または暗槓のデコード"""
    tile_id_raw = m >> 8
    tile_kind = tile_id_raw // 4
    
    if not (0 <= tile_kind <= 33):
        result.decode_error = f"槓: 無効な牌種 {tile_kind}"
        result.naki_type = NakiType.UNKNOWN
        return result
    
    base_id = tile_kind * 4
    
    # 4枚全てを取得
    tiles = []
    for i in range(4):
        tile_id = base_id + i
        if i == 0 and tile_kind in {4, 13, 22}:
            tile_id = {4: 16, 13: 52, 22: 88}[tile_kind]
        tiles.append(tile_id)
    
    result.tiles = sorted(tiles)
    
    # 大明槓 vs 暗槓の判定
    if from_who_relative != 0 and from_who_relative != 3:  # from_who=0は自分、3は上家だが暗槓でも0になることがある
        # 大明槓の可能性を検討
        # 暗槓の場合 from_who は通常 0 (自分自身扱い)
        if from_who_relative in {1, 2}:  # 対面または下家から
            result.naki_type = NakiType.DAIMINKAN
            result.consumed = []  # GameState で特定
        else:
            result.naki_type = NakiType.ANKAN
            result.from_who_relative = -1
            result.consumed = list(tiles)  # 4枚全て手牌から
    else:
        result.naki_type = NakiType.ANKAN
        result.from_who_relative = -1
        result.consumed = list(tiles)
    
    return result


# =============================================================================
# ユーティリティ関数
# =============================================================================

def get_naki_type_code(naki_type: str) -> int:
    """
    鳴きタイプ文字列から整数コードを取得（互換性用）
    
    Args:
        naki_type: "チー", "ポン", "大明槓", "加槓", "暗槓" のいずれか
    
    Returns:
        整数コード (0-4)、不明な場合は -1
    """
    nt = NAKI_TYPE_FROM_STRING.get(naki_type, NakiType.UNKNOWN)
    return NAKI_TYPE_CODES.get(nt, -1)


def naki_info_to_dict(naki_info: NakiInfo) -> Dict:
    """NakiInfoを辞書形式に変換（互換性用）"""
    return naki_info.to_dict()


def is_call_from_others(naki_type: NakiType) -> bool:
    """他家からの鳴きかどうか"""
    return naki_type in {NakiType.CHI, NakiType.PON, NakiType.DAIMINKAN}


def is_kan(naki_type: NakiType) -> bool:
    """槓かどうか（嶺上牌をツモる必要があるか）"""
    return naki_type in {NakiType.DAIMINKAN, NakiType.KAKAN, NakiType.ANKAN}


# =============================================================================
# 互換性のための関数（旧APIを維持）
# =============================================================================

# 旧コードで使われていた辞書形式の NAKI_TYPES
NAKI_TYPES = {
    "チー": 0,
    "ポン": 1,
    "大明槓": 2,
    "加槓": 3,
    "暗槓": 4,
    "不明": -1,
}


# =============================================================================
# テスト・検証
# =============================================================================

def run_self_tests() -> bool:
    """
    自己検査を実行
    """
    print("[naki_utils] 自己検査開始...")
    
    # テスト用のmコード（実際の天鳳ログから取得した値を使用）
    # 注意: これらの値は実際のログから確認する必要がある
    
    # 基本的なデコードテスト
    test_cases = [
        # (m_value, expected_type, description)
        # 実際のmコードがないため、型の一貫性だけをテスト
    ]
    
    # NakiInfoの基本テスト
    info = NakiInfo()
    assert info.naki_type == NakiType.UNKNOWN, "デフォルトタイプが不明でない"
    assert not info.is_valid(), "無効な情報が有効と判定された"
    
    # 辞書変換テスト
    info_dict = info.to_dict()
    assert "type" in info_dict, "to_dict()にtypeがない"
    assert "tiles" in info_dict, "to_dict()にtilesがない"
    
    # 型コード変換テスト
    assert get_naki_type_code("チー") == 0, "チーのコードが不正"
    assert get_naki_type_code("ポン") == 1, "ポンのコードが不正"
    assert get_naki_type_code("不明") == -1, "不明のコードが不正"
    
    print("[naki_utils] 全テスト通過")
    return True


if __name__ == "__main__":
    run_self_tests()
