# xml_parser.py
"""
天鳳XMLログパーサー

天鳳の牌譜XMLファイルを解析し、ゲームメタデータと局データを抽出する。

参考: https://blog.kobalab.net/entry/20170225/1488036549

XMLフォーマット概要:
- mjloggm: ルート要素 (ver属性でバージョン)
- SHUFFLE: 乱数シード情報
- GO: ルール情報 (赤あり、喰いあり等)
- UN: プレイヤー情報 (名前、段位、レート)
- TAIKYOKU: 対局開始 (起家)
- INIT: 局開始 (配牌、ドラ表示)
- T/U/V/W + 牌番号: ツモ (0-3番プレイヤー)
- D/E/F/G + 牌番号: 打牌 (大文字=手出し、小文字=ツモ切り)
- N: 副露 (m属性に面子コード)
- REACH: リーチ (step=1:宣言, step=2:確定)
- DORA: 新ドラ
- AGARI: 和了
- RYUUKYOKU: 流局
"""

import xml.etree.ElementTree as ET
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class GameMeta:
    """対局メタデータ"""
    # GO属性
    game_type: int = 0
    lobby: int = 0
    
    # 解釈済みルール
    is_sanma: bool = False           # 三麻
    is_tonpu: bool = False           # 東風戦 (Falseなら東南戦)
    has_aka: bool = True             # 赤あり
    has_kuitan: bool = True          # 喰いタンあり
    table_level: str = "一般"        # 卓レベル
    is_fast: bool = False            # 速卓
    
    # UN属性
    player_names: List[str] = field(default_factory=lambda: ["", "", "", ""])
    player_dans: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    player_rates: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    player_sexes: List[str] = field(default_factory=lambda: ["", "", "", ""])
    
    # TAIKYOKU
    first_dealer: int = 0


@dataclass
class RoundData:
    """局データ"""
    round_index: int = 0              # 局番号 (1-based)
    round_wind: int = 0               # 場風 (0:東, 1:南, 2:西)
    round_num: int = 0                # 何局目 (0-3)
    honba: int = 0                    # 本場
    kyotaku: int = 0                  # 供託
    dealer: int = 0                   # 親
    dora_indicator: int = -1          # 初期ドラ表示牌
    initial_scores: List[int] = field(default_factory=lambda: [25000, 25000, 25000, 25000])
    initial_hands: List[List[int]] = field(default_factory=lambda: [[], [], [], []])
    events: List[Dict] = field(default_factory=list)
    result: Optional[Dict] = None


# =============================================================================
# パーサー関数
# =============================================================================

def parse_go_type(type_value: int) -> Dict[str, Any]:
    """
    GO要素のtype属性を解析
    
    ビットフィールド:
    - 0x01: 人間との対戦
    - 0x02: 赤なし
    - 0x04: ナシナシ
    - 0x08: 東南戦
    - 0x10: 三麻
    - 0x20: 卓レベル(上位)
    - 0x40: 速卓
    - 0x80: 卓レベル(下位)
    
    卓レベル: 0x00=一般, 0x80=上級, 0x20=特上, 0xA0=鳳凰
    """
    result = {
        'is_vs_human': bool(type_value & 0x01),
        'has_aka': not bool(type_value & 0x02),
        'has_kuitan': not bool(type_value & 0x04),
        'is_tonpu': not bool(type_value & 0x08),  # 0=東風, 1=東南
        'is_sanma': bool(type_value & 0x10),
        'is_fast': bool(type_value & 0x40),
    }
    
    # 卓レベル
    level_bits = (type_value & 0x80) | (type_value & 0x20)
    if level_bits == 0x00:
        result['table_level'] = "一般"
    elif level_bits == 0x80:
        result['table_level'] = "上級"
    elif level_bits == 0x20:
        result['table_level'] = "特上"
    elif level_bits == 0xA0:
        result['table_level'] = "鳳凰"
    else:
        result['table_level'] = "不明"
    
    return result


def parse_init_seed(seed_str: str) -> Dict[str, int]:
    """
    INIT要素のseed属性を解析
    
    形式: "局順,本場,供託,サイコロ1,サイコロ2,ドラ表示牌"
    - 局順: 0=東1局, 1=東2局, ..., 4=南1局, ...
    - サイコロは実際の値-1 (0-5)
    """
    parts = seed_str.split(',')
    
    if len(parts) < 6:
        return {'round': 0, 'honba': 0, 'kyotaku': 0, 'dora': -1}
    
    try:
        round_value = int(parts[0])
        return {
            'round_wind': round_value // 4,  # 0=東, 1=南, 2=西
            'round_num': round_value % 4,    # 0-3
            'honba': int(parts[1]),
            'kyotaku': int(parts[2]),
            'dice1': int(parts[3]) + 1,      # 元の値に戻す
            'dice2': int(parts[4]) + 1,
            'dora': int(parts[5]),
        }
    except ValueError:
        return {'round_wind': 0, 'round_num': 0, 'honba': 0, 'kyotaku': 0, 'dora': -1}


def parse_tenhou_xml(xml_path: str) -> Tuple[GameMeta, List[RoundData]]:
    """
    天鳳XMLログファイルをパース
    
    Args:
        xml_path: XMLファイルのパス
    
    Returns:
        (GameMeta, List[RoundData])
    
    Raises:
        FileNotFoundError: ファイルが見つからない場合
        ET.ParseError: XML解析エラーの場合
    """
    xml_path = Path(xml_path)
    
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except ET.ParseError as e:
        raise ET.ParseError(f"Failed to parse XML: {xml_path} - {e}")
    
    meta = GameMeta()
    rounds = []
    current_round: Optional[RoundData] = None
    round_counter = 0
    
    for elem in root:
        tag = elem.tag
        attrib = elem.attrib
        
        # ===== メタデータ =====
        if tag == "GO":
            meta.game_type = int(attrib.get('type', 0))
            meta.lobby = int(attrib.get('lobby', 0))
            
            go_info = parse_go_type(meta.game_type)
            meta.has_aka = go_info['has_aka']
            meta.has_kuitan = go_info['has_kuitan']
            meta.is_tonpu = go_info['is_tonpu']
            meta.is_sanma = go_info['is_sanma']
            meta.is_fast = go_info['is_fast']
            meta.table_level = go_info['table_level']
        
        elif tag == "UN":
            # プレイヤー名 (URLエンコード)
            for i in range(4):
                name_key = f'n{i}'
                if name_key in attrib:
                    try:
                        meta.player_names[i] = urllib.parse.unquote(attrib[name_key])
                    except Exception:
                        meta.player_names[i] = f"player_{i}"
            
            # 段位
            if 'dan' in attrib:
                try:
                    meta.player_dans = [int(x) for x in attrib['dan'].split(',')]
                except ValueError:
                    pass
            
            # レート
            if 'rate' in attrib:
                try:
                    meta.player_rates = [float(x) for x in attrib['rate'].split(',')]
                except ValueError:
                    pass
            
            # 性別
            if 'sx' in attrib:
                meta.player_sexes = attrib['sx'].split(',')
        
        elif tag == "TAIKYOKU":
            meta.first_dealer = int(attrib.get('oya', 0))
        
        # ===== 局データ =====
        elif tag == "INIT":
            round_counter += 1
            current_round = RoundData(round_index=round_counter)
            
            # seed解析
            seed_info = parse_init_seed(attrib.get('seed', '0,0,0,0,0,0'))
            current_round.round_wind = seed_info.get('round_wind', 0)
            current_round.round_num = seed_info.get('round_num', 0)
            current_round.honba = seed_info.get('honba', 0)
            current_round.kyotaku = seed_info.get('kyotaku', 0)
            current_round.dora_indicator = seed_info.get('dora', -1)
            
            # 親
            current_round.dealer = int(attrib.get('oya', 0))
            
            # 初期点数
            if 'ten' in attrib:
                try:
                    current_round.initial_scores = [int(float(x)) * 100 for x in attrib['ten'].split(',')]
                except ValueError:
                    pass
            
            # 配牌
            for i in range(4):
                hai_key = f'hai{i}'
                if hai_key in attrib and attrib[hai_key]:
                    try:
                        current_round.initial_hands[i] = [int(x) for x in attrib[hai_key].split(',')]
                    except ValueError:
                        pass
            
            rounds.append(current_round)
        
        # ===== イベント =====
        elif current_round is not None:
            event = {'tag': tag, 'attrib': dict(attrib)}
            current_round.events.append(event)
            
            # 結果イベント
            if tag in ('AGARI', 'RYUUKYOKU'):
                current_round.result = event
    
    return meta, rounds


# =============================================================================
# 互換性のための関数
# =============================================================================

def parse_full_mahjong_log(xml_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    天鳳XMLログファイルをパース（ver_4.0.0互換形式）
    
    Args:
        xml_path: XMLファイルのパス
    
    Returns:
        (meta_dict, rounds_list)
    """
    try:
        meta, rounds = parse_tenhou_xml(xml_path)
    except (FileNotFoundError, ET.ParseError) as e:
        print(f"[Parser Error] {e}")
        return {}, []
    except Exception as e:
        print(f"[Parser Error] Unexpected error: {e}")
        return {}, []
    
    # メタデータを辞書に変換
    meta_dict = {
        'go': {'type': meta.game_type, 'lobby': meta.lobby},
        'player_names': meta.player_names,
        'has_aka': meta.has_aka,
        'has_kuitan': meta.has_kuitan,
        'is_sanma': meta.is_sanma,
        'table_level': meta.table_level,
    }
    
    # 局データを辞書リストに変換
    rounds_list = []
    for rd in rounds:
        round_dict = {
            'round_index': rd.round_index,
            'init': {
                'seed': f"{rd.round_wind * 4 + rd.round_num},{rd.honba},{rd.kyotaku},0,0,{rd.dora_indicator}",
                'ten': ','.join(str(s // 100) for s in rd.initial_scores),
                'oya': str(rd.dealer),
            },
            'events': rd.events,
            'result': rd.result,
        }
        
        # 配牌
        for i in range(4):
            if rd.initial_hands[i]:
                round_dict['init'][f'hai{i}'] = ','.join(str(t) for t in rd.initial_hands[i])
        
        rounds_list.append(round_dict)
    
    return meta_dict, rounds_list


# =============================================================================
# ユーティリティ
# =============================================================================

def get_xml_summary(xml_path: str) -> Dict[str, Any]:
    """
    XMLファイルのサマリーを取得
    
    Args:
        xml_path: XMLファイルのパス
    
    Returns:
        サマリー情報の辞書
    """
    try:
        meta, rounds = parse_tenhou_xml(xml_path)
    except Exception as e:
        return {'error': str(e)}
    
    return {
        'num_rounds': len(rounds),
        'players': meta.player_names,
        'has_aka': meta.has_aka,
        'table_level': meta.table_level,
        'is_sanma': meta.is_sanma,
        'rounds': [
            {
                'round': f"{'東南西'[r.round_wind]}{r.round_num + 1}局",
                'honba': r.honba,
                'num_events': len(r.events),
                'result': r.result['tag'] if r.result else None,
            }
            for r in rounds
        ]
    }


# =============================================================================
# テスト
# =============================================================================

def run_self_tests() -> bool:
    """自己検査"""
    print("[xml_parser] 自己検査開始...")
    
    # GO type解析テスト
    # 四特南喰赤 = 0x29 = 41
    go_info = parse_go_type(41)
    assert go_info['has_aka'] == True
    assert go_info['has_kuitan'] == True
    assert go_info['is_tonpu'] == False  # 東南戦
    assert go_info['table_level'] == "特上"
    
    # seed解析テスト
    seed_info = parse_init_seed("6,1,2,1,3,16")
    assert seed_info['round_wind'] == 1  # 南
    assert seed_info['round_num'] == 2   # 3局目
    assert seed_info['honba'] == 1
    assert seed_info['kyotaku'] == 2
    assert seed_info['dora'] == 16
    
    print("[xml_parser] 全テスト通過")
    return True


if __name__ == "__main__":
    run_self_tests()


