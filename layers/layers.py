# kehanet/layers/base.py
from __future__ import annotations
from typing import List, Protocol
from core.tensor import Tensor

class LayerProtocol(Protocol):
    """
    Katmanların uyması gereken arayüz.
    """
    training: bool
    def __call__(self, x: Tensor) -> Tensor: ...
    def parameters(self) -> List[Tensor]: ...
    def zero_grad(self) -> None: ...

class Layer:
    """
    Tüm katmanların temel sınıfı.

    Ortak işlevler:
      - forward     : Alt sınıfta override edilmeli
      - __call__    : forward delegasyonu
      - parameters  : Öğrenilebilir tensörleri döner
      - zero_grad   : Parametre gradyanlarını temizler
      - train / eval: Eğitim/değerlendirme modunu ayarlar
    """
    def __init__(self) -> None:
        self._params: List[Tensor] = []
        self.training: bool = True

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError(f"forward not implemented in {self.__class__.__name__}")

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        return list(self._params)

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad = None

    def train(self) -> Layer:
        self.training = True
        return self

    def eval(self) -> Layer:
        self.training = False
        return self

    def __repr__(self) -> str:
        params = ", ".join(repr(p) for p in self._params)
        return f"{self.__class__.__name__}(training={self.training}, params=[{params}])"


# kehanet/layers/dense.py
import numpy as np
from typing import Callable, Optional
from core.tensor import Tensor
from base import Layer

class Dense(Layer):
    """
    Tam bağlantılı katman: y = activation(x @ W + b);
    dropout eğitim modunda uygulanır.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[Callable[[Tensor], Tensor]] = None,
        bias: bool = True,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = Tensor(
            np.random.uniform(-limit, limit, size=(in_features, out_features)),
            requires_grad=True
        )
        self.b = (
            Tensor(np.zeros((1, out_features)), requires_grad=True)
            if bias else None
        )
        self.activation = activation
        self.dropout = float(dropout)
        self._params = [self.W] + ([self.b] if self.b is not None else [])

    def forward(self, x: Tensor) -> Tensor:
        z = x @ self.W
        if self.b is not None:
            z = z + self.b
        if self.activation:
            z = self.activation(z)
        # dropout only in training
        if self.dropout > 0.0 and self.training:
            mask = (np.random.rand(*z.shape) > self.dropout).astype(np.float32)
            z = z * Tensor(mask, requires_grad=False)
        return z

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base[:-1]}, dropout={self.dropout})"