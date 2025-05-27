# core/optimizers/sgd.py
"""
Stokastik Gradyan İnişi (SGD) optimizasyonu.
"""
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import numpy as np
from core.tensor import Tensor

class SGD:
    """
    Basit SGD optimizer.

    Attributes
    ----------
    params : List[Tensor]
        Optimize edilecek parametreler.
    lr : float
        Öğrenme oranı.
    momentum : float
        Momentum katsayısı (0 ise momentum kullanılmaz).
    weight_decay : float
        Ağırlık çürümesi katsayısı.
    nesterov : bool
        Nesterov momentum kullanılsın mı?
    """
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ) -> None:
        self.params: List[Tensor] = list(params)
        self.lr: float = lr
        self.momentum: float = momentum
        self.weight_decay: float = weight_decay
        self.nesterov: bool = nesterov
        # Momentum buffer
        self._velocity: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """
        Bir adım güncelleme: p = p - lr * grad (+ momentum, weight decay).
        """
        for idx, p in enumerate(self.params):
            grad = p.grad
            if grad is None:
                continue
            # weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            # momentum update
            if self.momentum != 0:
                v = self._velocity[idx]
                v_new = self.momentum * v + grad
                self._velocity[idx] = v_new
                if self.nesterov:
                    update = grad + self.momentum * v_new
                else:
                    update = v_new
            else:
                update = grad
            # parameter update
            p.data = p.data - self.lr * update

    def zero_grad(self) -> None:
        """
        Tüm parametrelerin gradyanlarını temizler.
        """
        for p in self.params:
            p.grad = None

    def __repr__(self) -> str:
        return (
            f"SGD(lr={self.lr}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay}, nesterov={self.nesterov})"
        )
