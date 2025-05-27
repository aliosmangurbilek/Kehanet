# kehanet/layers/dense.py
import numpy as np
from typing import Callable, Optional
from core.tensor import Tensor
from layers.base import Layer

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