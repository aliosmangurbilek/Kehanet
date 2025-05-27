# core/data/dataloader.py
from __future__ import annotations
import gzip
import struct
from pathlib import Path
import numpy as np
from typing import Tuple, Iterator, Optional, Union
from core.tensor import Tensor

class Dataset:
    """
    Ham veri ve etiketleri taşıyan basit veri kümesi sınıfı.
    """
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ):
        assert data.shape[0] == labels.shape[0], "Veri ve etiket sayıları uyuşmuyor"
        self.data: np.ndarray = data.astype(np.float32)
        self.labels: np.ndarray = labels.astype(np.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.data[idx], self.labels[idx]

class DataLoader:
    """
    Mini-batch veri yükleyici.

    Yields X_batch, y_batch (numpy dizileri veya Tensorler).
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        to_tensor: bool = False
    ):
        self.dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.to_tensor: bool = to_tensor
        self._indices: np.ndarray = np.arange(len(dataset))
        self._pos: int = 0

    def __iter__(self) -> DataLoader:
        if self.shuffle:
            np.random.shuffle(self._indices)
        self._pos = 0
        return self

    def __next__(self) -> Tuple[Union[np.ndarray, Tensor], Union[np.ndarray, Tensor]]:
        if self._pos >= len(self._indices):
            raise StopIteration
        start = self._pos
        end = min(start + self.batch_size, len(self._indices))
        idx = self._indices[start:end]
        X_batch = self.dataset.data[idx]
        y_batch = self.dataset.labels[idx]
        self._pos = end
        if self.to_tensor:
            X_batch = Tensor(X_batch)
            y_batch = Tensor(y_batch)
        return X_batch, y_batch

# --- Yardımcı fonksiyonlar ---

def _resolve_path(path: Path) -> Path:
    """
    .gz ile sıkıştırılmış dosya varsa onu, yoksa orijinal dosyayı döner.
    """
    gz_path = path.with_suffix(path.suffix + '.gz')
    if gz_path.exists():
        return gz_path
    if path.exists():
        return path
    raise FileNotFoundError(f"Dosya bulunamadı: {path}")

def _read_idx(path: Path) -> np.ndarray:
    """
    IDX formatındaki (sıkıştırılmış veya düz) dosyayı numpy dizisine çevirir.
    """
    open_fn = gzip.open if path.suffix == '.gz' else open
    with open_fn(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic in (0x00000803, 2051):  # image file
            num, rows, cols = struct.unpack('>III', f.read(12))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num, rows * cols)
        if magic in (0x00000801, 2049):  # label file
            num, = struct.unpack('>I', f.read(4))
            return np.frombuffer(f.read(), dtype=np.uint8)
        raise ValueError(f"Geçersiz IDX magic numarası: {magic}")


def load_mnist_local(
    folder: Union[str, Path] = 'datasets/mnist',
    normalize: bool = True,
    one_hot: bool = True
) -> Tuple[Dataset, Dataset]:
    """
    MNIST veri setini folder altında arayıp yükler.
    normalize: True ise [0,1] aralığına ölçekler.
    one_hot: True ise etiketleri one-hot kodlamasına çevirir.
    """
    folder = Path(folder)
    # Dosya yollarını çöz ve oku
    img_train = _read_idx(_resolve_path(folder / 'train-images.idx3-ubyte'))
    y_train   = _read_idx(_resolve_path(folder / 'train-labels.idx1-ubyte'))
    img_test  = _read_idx(_resolve_path(folder / 't10k-images.idx3-ubyte'))
    y_test    = _read_idx(_resolve_path(folder / 't10k-labels.idx1-ubyte'))

    # Normalize
    if normalize:
        X_train = img_train.astype(np.float32) / 255.0
        X_test  = img_test.astype(np.float32) / 255.0
    else:
        X_train = img_train.astype(np.float32)
        X_test  = img_test.astype(np.float32)

    # Etiketleri hazırla
    if one_hot:
        def to_one_hot(y: np.ndarray) -> np.ndarray:
            n = y.shape[0]
            oh = np.zeros((n, 10), dtype=np.float32)
            oh[np.arange(n), y] = 1.0
            return oh
        y_train_ = to_one_hot(y_train)
        y_test_  = to_one_hot(y_test)
    else:
        y_train_ = y_train.astype(np.float32)
        y_test_  = y_test.astype(np.float32)

    return Dataset(X_train, y_train_), Dataset(X_test, y_test_)
