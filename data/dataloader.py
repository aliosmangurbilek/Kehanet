# kehanet/data/dataloader.py
"""
Local MNIST veri seti yükleyici ve DataLoader sınıfı.
Veri dosyaları `datasets/mnist` dizininde `.idx3-ubyte` ve `.idx1-ubyte` formatında bulunmalıdır.
Girilen dosya sıkıştırılmış (.gz) ya da düz formatta olabilir.
"""
import gzip
import struct
import os
import numpy as np
from typing import Tuple, Iterator
from core.tensor import Tensor

class Dataset:
    """
    Basit Dataset sınıfı: ham veri ve etiketleri taşır.
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        assert data.shape[0] == labels.shape[0], "Veri ve etiket sayıları uyuşmuyor"
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.data[idx], self.labels[idx]

class DataLoader:
    """
    Mini-batch veri yükleyici.
    Kullanım: for X_batch, y_batch in DataLoader(dataset, batch_size=32):
    """
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = np.arange(len(dataset))
        self._position = 0

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.shuffle:
            np.random.shuffle(self._indices)
        self._position = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._position >= len(self._indices):
            raise StopIteration
        start = self._position
        end = min(start + self.batch_size, len(self._indices))
        idx = self._indices[start:end]
        batch_data = self.dataset.data[idx]
        batch_labels = self.dataset.labels[idx]
        self._position = end
        return batch_data, batch_labels


def _read_idx(path: str) -> np.ndarray:
    """
    IDX formatındaki dosyayı (.gz veya düz) okuyup numpy array olarak döner.
    """
    open_fn = gzip.open if path.endswith('.gz') else open
    with open_fn(path, 'rb') as f:
        # İlk 4 byte magic number
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number == 0x00000803 or magic_number == 2051:
            # images
            num, rows, cols = struct.unpack('>III', f.read(12))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num, rows * cols)
        elif magic_number == 0x00000801 or magic_number == 2049:
            # labels
            num, = struct.unpack('>I', f.read(4))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data
        else:
            raise ValueError(f"Geçersiz IDX magic numarası: {magic_number}")


def load_mnist_local(
    folder: str = 'datasets/mnist',
    normalize: bool = True,
    one_hot: bool = True
) -> Tuple[Dataset, Dataset]:
    """
    Local MNIST veri setini yükler.
    folder: MNIST dosyalarının bulunduğu dizin (path/to/datasets/mnist)
    normalize: True ise [0,1] aralığına ölçekler
    one_hot: True ise etiketleri one-hot formata çevirir

    Gerekli dosyalar:
        train-images-idx3-ubyte[.gz]
        train-labels-idx1-ubyte[.gz]
        t10k-images-idx3-ubyte[.gz]
        t10k-labels-idx1-ubyte[.gz]
    """
    # Dosya isimleri
    img_train = os.path.join(folder, 'train-images.idx3-ubyte')
    lbl_train = os.path.join(folder, 'train-labels.idx1-ubyte')
    img_test  = os.path.join(folder, 't10k-images.idx3-ubyte')
    lbl_test  = os.path.join(folder, 't10k-labels.idx1-ubyte')

    # Eğer .gz uzantılı hali varsa, ona yönlendir
    def _choose(path):
        if os.path.exists(path + '.gz'):
            return path + '.gz'
        return path

    img_train = _choose(img_train)
    lbl_train = _choose(lbl_train)
    img_test  = _choose(img_test)
    lbl_test  = _choose(lbl_test)

    # Verileri oku
    X_train = _read_idx(img_train)
    y_train = _read_idx(lbl_train)
    X_test  = _read_idx(img_test)
    y_test  = _read_idx(lbl_test)

    # Normalizasyon
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test  = X_test.astype(np.float32) / 255.0
    else:
        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)

    # Etiket formatı
    if one_hot:
        n_train = y_train.shape[0]
        n_test  = y_test.shape[0]
        labels_train = np.zeros((n_train, 10), dtype=np.float32)
        labels_train[np.arange(n_train), y_train] = 1.0
        labels_test  = np.zeros((n_test, 10), dtype=np.float32)
        labels_test[np.arange(n_test), y_test]   = 1.0
    else:
        labels_train = y_train.astype(np.float32)
        labels_test  = y_test.astype(np.float32)

    return Dataset(X_train, labels_train), Dataset(X_test, labels_test)

def load_mnist_local(
    folder: str = 'datasets/mnist',
    normalize: bool = True,
    one_hot: bool = True
) -> Tuple[Dataset, Dataset]:
    """
    Local MNIST veri setini yükler.
    folder: MNIST dosyalarının bulunduğu dizin (path/to/datasets/mnist)
    normalize: True ise [0,1] aralığına ölçekler
    one_hot: True ise etiketleri one-hot formata çevirir

    Gerekli dosyalar:
        train-images-idx3-ubyte[.gz]
        train-labels-idx1-ubyte[.gz]
        t10k-images-idx3-ubyte[.gz]
        t10k-labels-idx1-ubyte[.gz]
    """
    # Yol tanımları
    img_train = os.path.join(folder, 'train-images.idx3-ubyte')
    lbl_train = os.path.join(folder, 'train-labels.idx1-ubyte')
    img_test  = os.path.join(folder, 't10k-images.idx3-ubyte')
    lbl_test  = os.path.join(folder, 't10k-labels.idx1-ubyte')
    # varsa .gz eke bak
    for p in [img_train, lbl_train, img_test, lbl_test]:
        if not os.path.exists(p) and os.path.exists(p):
            p += '.gz'
    # Oku
    X_train = _read_idx(img_train if os.path.exists(img_train) else img_train )
    y_train = _read_idx(lbl_train if os.path.exists(lbl_train) else lbl_train )
    X_test  = _read_idx(img_test  if os.path.exists(img_test)  else img_test  )
    y_test  = _read_idx(lbl_test  if os.path.exists(lbl_test)  else lbl_test  )

    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test  = X_test.astype(np.float32) / 255.0
    else:
        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)

    if one_hot:
        n_train = y_train.shape[0]
        n_test  = y_test.shape[0]
        labels_train = np.zeros((n_train, 10), dtype=np.float32)
        labels_train[np.arange(n_train), y_train] = 1.0
        labels_test  = np.zeros((n_test, 10), dtype=np.float32)
        labels_test[np.arange(n_test), y_test]   = 1.0
    else:
        labels_train = y_train.astype(np.float32)
        labels_test  = y_test.astype(np.float32)

    return Dataset(X_train, labels_train), Dataset(X_test, labels_test)