# utils/__init__.py
"""
麻雀AI ユーティリティパッケージ

使用例:
    from utils import tile_id_to_kind, is_aka_dora
    from utils.dataset_utils import NpzDataset, create_data_loaders
    from utils.feature_utils import hand_to_count, create_valid_action_mask
    from utils.xml_parser import parse_tenhou_xml
"""

# 牌ユーティリティ
from .tile_utils import (
    # 定数
    AKA_MAN_ID, AKA_PIN_ID, AKA_SOU_ID, AKA_DORA_IDS,
    NUM_TILE_TYPES, TILE_ID_MIN, TILE_ID_MAX,
    Suit, Honor,
    TILE_NAMES_SHORT, HONOR_NAMES,
    TileInfo,
    
    # 基本関数
    is_valid_tile_id, is_valid_tile_kind, is_aka_dora,
    tile_id_to_kind, tile_id_to_index,  # エイリアス
    tile_kind_to_id, tile_index_to_id,  # エイリアス
    tile_id_to_string, tile_kind_to_string,
    get_tile_info,
    
    # 手牌操作
    hand_to_kinds, hand_to_string, count_tiles_by_kind,
    count_aka_dora, find_tiles_by_kind,
    is_tile_in_hand, is_kind_in_hand,
    
    # ドラ
    dora_indicator_to_dora,
    
    # 検証
    validate_hand, debug_print_hand,
    
    # 特徴量
    hand_to_34_array, hand_to_34_with_aka,
)

# 鳴きユーティリティ
from .naki_utils import (
    NakiType, NakiInfo,
    NAKI_TYPE_CODES, NAKI_TYPES,
    decode_naki,
    get_naki_type_code, is_call_from_others, is_kan,
)

# XMLパーサー
from .xml_parser import (
    GameMeta, RoundData,
    parse_tenhou_xml, parse_full_mahjong_log,
    parse_go_type, parse_init_seed,
    get_xml_summary,
)

# 局面状態（完全情報。モデル入力には観測専用builderを使用する）
from .game_state import GameState, MAX_EVENT_HISTORY, STATIC_FEATURE_DIM

# 特徴量ユーティリティ
from .feature_utils import (
    hand_to_one_hot, hand_to_count, hand_to_count_with_aka,
    create_valid_action_mask,
    river_to_sequence, river_to_count,
    visible_tiles_count, remaining_tiles_count,
    dora_count_in_hand, aka_dora_count_in_hand,
    dora_indicator_features, dora_tiles_features,
    player_position_features, riichi_status_features,
    create_full_features,
)

# データセットユーティリティ
from .dataset_utils import (
    NpzDataset,
    load_dataset, combine_datasets,
    get_class_weights,
)

# PyTorch関連（オプション）
try:
    from .dataset_utils import (
        MahjongTorchDataset,
        create_data_loaders,
    )
except ImportError:
    pass  # PyTorchがない場合はスキップ


__all__ = [
    # tile_utils
    'AKA_MAN_ID', 'AKA_PIN_ID', 'AKA_SOU_ID', 'AKA_DORA_IDS',
    'NUM_TILE_TYPES', 'TILE_ID_MIN', 'TILE_ID_MAX',
    'Suit', 'Honor', 'TILE_NAMES_SHORT', 'HONOR_NAMES', 'TileInfo',
    'is_valid_tile_id', 'is_valid_tile_kind', 'is_aka_dora',
    'tile_id_to_kind', 'tile_id_to_index',
    'tile_kind_to_id', 'tile_index_to_id',
    'tile_id_to_string', 'tile_kind_to_string',
    'get_tile_info',
    'hand_to_kinds', 'hand_to_string', 'count_tiles_by_kind',
    'count_aka_dora', 'find_tiles_by_kind',
    'is_tile_in_hand', 'is_kind_in_hand',
    'dora_indicator_to_dora',
    'validate_hand', 'debug_print_hand',
    'hand_to_34_array', 'hand_to_34_with_aka',
    
    # naki_utils
    'NakiType', 'NakiInfo',
    'NAKI_TYPE_CODES', 'NAKI_TYPES',
    'decode_naki', 'get_naki_type_code', 'is_call_from_others', 'is_kan',
    
    # xml_parser
    'GameMeta', 'RoundData',
    'parse_tenhou_xml', 'parse_full_mahjong_log',
    'parse_go_type', 'parse_init_seed', 'get_xml_summary',
    'GameState', 'MAX_EVENT_HISTORY', 'STATIC_FEATURE_DIM',
    
    # feature_utils
    'hand_to_one_hot', 'hand_to_count', 'hand_to_count_with_aka',
    'create_valid_action_mask',
    'river_to_sequence', 'river_to_count',
    'visible_tiles_count', 'remaining_tiles_count',
    'dora_count_in_hand', 'aka_dora_count_in_hand',
    'dora_indicator_features', 'dora_tiles_features',
    'player_position_features', 'riichi_status_features',
    'create_full_features',
    
    # dataset_utils
    'NpzDataset', 'load_dataset', 'combine_datasets', 'get_class_weights',
]


