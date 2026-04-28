# dataset_utils.py
"""
学習データセットのユーティリティ

NPZ/HDF5形式のデータセット読み込み、バッチ処理、
PyTorch/NumPy形式への変換などを提供。
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Iterator, Union
import numpy as np

# PyTorchのインポート（オプション）
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# HDF5のインポート（オプション）
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


# =============================================================================
# NPZデータセット
# =============================================================================

class NpzDataset:
    """
    NPZ形式のデータセットを読み込むクラス
    
    使用例:
        dataset = NpzDataset("./dataset/train.npz")
        print(f"サンプル数: {len(dataset)}")
        
        # インデックスアクセス
        sample = dataset[0]
        
        # イテレーション
        for batch in dataset.batch_iter(batch_size=32):
            ...
    """
    
    def __init__(self, npz_path: str, load_to_memory: bool = True):
        """
        Args:
            npz_path: NPZファイルのパス
            load_to_memory: Trueの場合、全データをメモリにロード
        """
        self.npz_path = Path(npz_path)
        
        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        self._data = np.load(str(self.npz_path), allow_pickle=True)
        
        # 利用可能なキーを確認
        self._keys = list(self._data.keys())
        
        # メインデータ
        self.static_features = self._data.get('static_features', None)
        self.sequence_features = self._data.get('sequence_features', None)
        self.hand_counts = self._data.get('hand_counts', None)
        self.aka_flags = self._data.get('aka_flags', None)
        self.valid_masks = self._data.get('valid_masks', None)
        self.labels = self._data.get('labels', None)
        
        # メタデータ
        self.file_ids = self._data.get('file_ids', None)
        self.round_indices = self._data.get('round_indices', None)
        self.junmes = self._data.get('junmes', None)
        self.player_ids = self._data.get('player_ids', None)
        
        # サンプル数
        self._length = len(self.labels) if self.labels is not None else 0
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        インデックスでサンプルを取得
        
        Returns:
            サンプルの辞書
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length})")
        
        sample = {
            'static': self.static_features[idx],
            'sequence': self.sequence_features[idx],
            'hand_counts': self.hand_counts[idx],
            'aka_flags': self.aka_flags[idx],
            'valid_mask': self.valid_masks[idx] if self.valid_masks is not None else None,
            'label': self.labels[idx],
        }
        
        # メタデータがあれば追加
        if self.file_ids is not None:
            sample['file_id'] = self.file_ids[idx]
        if self.round_indices is not None:
            sample['round_index'] = self.round_indices[idx]
        if self.junmes is not None:
            sample['junme'] = self.junmes[idx]
        if self.player_ids is not None:
            sample['player_id'] = self.player_ids[idx]
        
        return sample
    
    def get_batch(self, indices: List[int]) -> Dict[str, np.ndarray]:
        """
        複数インデックスのバッチを取得
        """
        return {
            'static': self.static_features[indices],
            'sequence': self.sequence_features[indices],
            'hand_counts': self.hand_counts[indices],
            'aka_flags': self.aka_flags[indices],
            'valid_masks': self.valid_masks[indices] if self.valid_masks is not None else None,
            'labels': self.labels[indices],
        }
    
    def batch_iter(
        self, 
        batch_size: int, 
        shuffle: bool = False, 
        seed: Optional[int] = None
    ) -> Iterator[Dict[str, np.ndarray]]:
        """
        バッチイテレータ
        
        Args:
            batch_size: バッチサイズ
            shuffle: シャッフルするかどうか
            seed: シャッフル用の乱数シード
        
        Yields:
            バッチの辞書
        """
        indices = np.arange(self._length)
        
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        
        for start in range(0, self._length, batch_size):
            end = min(start + batch_size, self._length)
            batch_indices = indices[start:end]
            yield self.get_batch(batch_indices)
    
    def get_label_distribution(self) -> Dict[int, int]:
        """ラベル分布を取得"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_statistics(self) -> Dict:
        """データセットの統計情報を取得"""
        return {
            'num_samples': self._length,
            'static_shape': self.static_features.shape if self.static_features is not None else None,
            'sequence_shape': self.sequence_features.shape if self.sequence_features is not None else None,
            'hand_shape': self.hand_counts.shape if self.hand_counts is not None else None,
            'valid_mask_shape': self.valid_masks.shape if self.valid_masks is not None else None,
            'num_labels': len(np.unique(self.labels)) if self.labels is not None else 0,
            'label_distribution': self.get_label_distribution(),
        }
    
    def close(self):
        """リソースを解放"""
        if hasattr(self._data, 'close'):
            self._data.close()


# =============================================================================
# PyTorchデータセット
# =============================================================================

if TORCH_AVAILABLE:
    class MahjongTorchDataset(Dataset):
        """
        PyTorch用のデータセット
        
        使用例:
            dataset = MahjongTorchDataset("./dataset/train.npz")
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            for batch in loader:
                static = batch['static']
                labels = batch['label']
                ...
        """
        
        def __init__(
            self, 
            npz_path: str,
            use_sequence: bool = True,
            use_hand_counts: bool = True,
            use_aka_flags: bool = True,
            device: Optional[str] = None
        ):
            """
            Args:
                npz_path: NPZファイルのパス
                use_sequence: イベント系列を使用するか
                use_hand_counts: 手牌カウントを使用するか
                use_aka_flags: 赤ドラフラグを使用するか
                device: デバイス指定（'cuda', 'cpu', None=自動）
            """
            data = np.load(npz_path, allow_pickle=True)
            
            self.static_features = torch.tensor(data['static_features'], dtype=torch.float32)
            self.labels = torch.tensor(data['labels'], dtype=torch.long)
            
            self.use_sequence = use_sequence
            self.use_hand_counts = use_hand_counts
            self.use_aka_flags = use_aka_flags
            
            if use_sequence:
                self.sequence_features = torch.tensor(data['sequence_features'], dtype=torch.float32)
            if use_hand_counts:
                self.hand_counts = torch.tensor(data['hand_counts'], dtype=torch.float32)
            if use_aka_flags:
                self.aka_flags = torch.tensor(data['aka_flags'], dtype=torch.float32)
            self.valid_masks = torch.tensor(
                data['valid_masks'] if 'valid_masks' in data else np.ones((len(self.labels), 34), dtype=np.float32),
                dtype=torch.float32,
            )
            
            self._length = len(self.labels)
            
            # デバイスへの転送
            if device:
                self.to(device)
        
        def __len__(self) -> int:
            return self._length
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            sample = {
                'static': self.static_features[idx],
                'label': self.labels[idx],
            }
            
            if self.use_sequence:
                sample['sequence'] = self.sequence_features[idx]
            if self.use_hand_counts:
                sample['hand_counts'] = self.hand_counts[idx]
            if self.use_aka_flags:
                sample['aka_flags'] = self.aka_flags[idx]
            sample['valid_mask'] = self.valid_masks[idx]
            
            return sample
        
        def to(self, device: str) -> 'MahjongTorchDataset':
            """データをデバイスに転送"""
            self.static_features = self.static_features.to(device)
            self.labels = self.labels.to(device)
            
            if self.use_sequence:
                self.sequence_features = self.sequence_features.to(device)
            if self.use_hand_counts:
                self.hand_counts = self.hand_counts.to(device)
            if self.use_aka_flags:
                self.aka_flags = self.aka_flags.to(device)
            self.valid_masks = self.valid_masks.to(device)
            
            return self
        
        def get_input_dims(self) -> Dict[str, int]:
            """各入力の次元を取得"""
            dims = {
                'static_dim': self.static_features.shape[1],
            }
            
            if self.use_sequence:
                dims['sequence_length'] = self.sequence_features.shape[1]
                dims['sequence_dim'] = self.sequence_features.shape[2]
            if self.use_hand_counts:
                dims['hand_dim'] = self.hand_counts.shape[1]
            if self.use_aka_flags:
                dims['aka_dim'] = self.aka_flags.shape[1]
            dims['valid_mask_dim'] = self.valid_masks.shape[1]
            
            return dims


# =============================================================================
# HDF5データセット
# =============================================================================

if HDF5_AVAILABLE:
    class Hdf5Dataset:
        """
        HDF5形式のデータセットを読み込むクラス
        
        大規模データセット向け。メモリにロードせずにアクセス可能。
        """
        
        def __init__(self, hdf5_path: str, mode: str = 'r'):
            """
            Args:
                hdf5_path: HDF5ファイルのパス
                mode: 'r'=読み取り専用, 'r+'=読み書き
            """
            self.hdf5_path = Path(hdf5_path)
            
            if not self.hdf5_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
            
            self._file = h5py.File(str(self.hdf5_path), mode)
            
            # データセットの参照を取得
            self.static_features = self._file.get('static_features', None)
            self.sequence_features = self._file.get('sequences', None)
            self.labels = self._file.get('labels', None)
            
            self._length = len(self.labels) if self.labels is not None else 0
        
        def __len__(self) -> int:
            return self._length
        
        def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
            if idx < 0 or idx >= self._length:
                raise IndexError(f"Index {idx} out of range")
            
            return {
                'static': self.static_features[idx] if self.static_features else None,
                'sequence': self.sequence_features[idx] if self.sequence_features else None,
                'label': self.labels[idx] if self.labels else None,
            }
        
        def get_batch(self, start: int, end: int) -> Dict[str, np.ndarray]:
            """スライスでバッチを取得"""
            return {
                'static': self.static_features[start:end] if self.static_features else None,
                'sequence': self.sequence_features[start:end] if self.sequence_features else None,
                'labels': self.labels[start:end] if self.labels else None,
            }
        
        def close(self):
            """ファイルを閉じる"""
            self._file.close()
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()


# =============================================================================
# ユーティリティ関数
# =============================================================================

def load_dataset(path: str) -> Union[NpzDataset, 'Hdf5Dataset']:
    """
    パスから適切なデータセットクラスを選択してロード
    
    Args:
        path: データセットファイルのパス (.npz or .hdf5)
    
    Returns:
        データセットオブジェクト
    """
    path = Path(path)
    
    if path.suffix == '.npz':
        return NpzDataset(str(path))
    elif path.suffix in ('.hdf5', '.h5'):
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 files")
        return Hdf5Dataset(str(path))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def combine_datasets(*datasets: NpzDataset) -> Dict[str, np.ndarray]:
    """
    複数のNpzDatasetを結合
    
    Args:
        *datasets: 結合するデータセット
    
    Returns:
        結合されたデータの辞書
    """
    if not datasets:
        return {}
    
    static_list = [d.static_features for d in datasets if d.static_features is not None]
    sequence_list = [d.sequence_features for d in datasets if d.sequence_features is not None]
    hand_list = [d.hand_counts for d in datasets if d.hand_counts is not None]
    aka_list = [d.aka_flags for d in datasets if d.aka_flags is not None]
    label_list = [d.labels for d in datasets if d.labels is not None]
    
    return {
        'static_features': np.concatenate(static_list) if static_list else None,
        'sequence_features': np.concatenate(sequence_list) if sequence_list else None,
        'hand_counts': np.concatenate(hand_list) if hand_list else None,
        'aka_flags': np.concatenate(aka_list) if aka_list else None,
        'labels': np.concatenate(label_list) if label_list else None,
    }


def get_class_weights(labels: np.ndarray, num_classes: int = 34) -> np.ndarray:
    """
    クラス重みを計算（不均衡データ対策）
    
    Args:
        labels: ラベル配列
        num_classes: クラス数
    
    Returns:
        クラス重みの配列
    """
    counts = np.bincount(labels, minlength=num_classes)
    # 0除算回避
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    # 正規化
    weights = weights / weights.sum() * num_classes
    return weights.astype(np.float32)


def create_data_loaders(
    train_path: str,
    valid_path: Optional[str] = None,
    test_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
    **dataset_kwargs
) -> Dict[str, 'DataLoader']:
    """
    DataLoaderを作成（PyTorch用）
    
    Args:
        train_path: 訓練データのパス
        valid_path: 検証データのパス（オプション）
        test_path: テストデータのパス（オプション）
        batch_size: バッチサイズ
        num_workers: データロードのワーカー数
        shuffle_train: 訓練データをシャッフルするか
        **dataset_kwargs: MahjongTorchDatasetに渡す追加引数
    
    Returns:
        {"train": loader, "valid": loader, "test": loader} の辞書
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for create_data_loaders")
    
    loaders = {}
    
    train_dataset = MahjongTorchDataset(train_path, **dataset_kwargs)
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    
    if valid_path and Path(valid_path).exists():
        valid_dataset = MahjongTorchDataset(valid_path, **dataset_kwargs)
        loaders['valid'] = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    if test_path and Path(test_path).exists():
        test_dataset = MahjongTorchDataset(test_path, **dataset_kwargs)
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    return loaders


# =============================================================================
# テスト
# =============================================================================

def run_self_tests() -> bool:
    """自己検査"""
    print("[dataset_utils] 自己検査開始...")
    
    # クラス重みテスト
    labels = np.array([0, 0, 1, 1, 1, 2])
    weights = get_class_weights(labels, num_classes=3)
    assert len(weights) == 3
    assert weights.sum() > 0
    
    print("[dataset_utils] 全テスト通過")
    return True


if __name__ == "__main__":
    run_self_tests()


