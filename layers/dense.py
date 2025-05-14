# kehanet/layers/dense.py
"""
Dense (Tam Bağlantılı) Katmanı
==============================

y = activation(x @ W + b)
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional

from core.tensor import Tensor
from layers.base import Layer


class Dense(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[Callable[[Tensor], Tensor]] = None,
        bias: bool = True,
    ):
        super().__init__()
        # Xavier/Glorot benzeri küçük başlangıç
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = Tensor(
            np.random.uniform(-limit, limit, size=(in_features, out_features)),
            requires_grad=True,  # Set requires_grad=True for W
        )
        self.use_bias = bias
        self.b = (
            Tensor(np.zeros((1, out_features), dtype=np.float32), requires_grad=True)  # Set requires_grad=True for b
            if bias
            else None
        )
        self.activation = activation  # örn: ag.relu, ag.sigmoid
        # parametre listesini güncelle
        self._params = [self.W] + ([self.b] if self.use_bias else [])

    # ---------------------------------------------------------------------- #
    def forward(self, x: Tensor) -> Tensor:
        z = x @ self.W
        if self.use_bias:
            z = z + self.b
        if self.activation is not None:
            z = self.activation(z)
        z.requires_grad = x.requires_grad or self.W.requires_grad or (self.b.requires_grad if self.b is not None else False)
        return z

