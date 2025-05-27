# core/optimizers/adam.py
"""
Adaptive Moment Estimation (Adam) optimizer implementation.
Kingma & Ba, 2014.
https://arxiv.org/abs/1412.6980
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from core.tensor import Tensor

class Adam:
    """
    Adam optimizer.

    Attributes
    ----------
    parameters : List[Tensor]
        Optimize edilecek parametreler.
    lr : float
        Öğrenme oranı.
    betas : Tuple[float, float]
        (beta1, beta2) moment katsayıları.
    eps : float
        Sayısal kararlılık epsilonu.
    """
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8
    ) -> None:
        self.parameters: List[Tensor] = parameters
        self.lr: float = lr
        self.beta1, self.beta2 = betas
        self.eps: float = eps
        # İlk ve ikinci momentler
        self._m: List[np.ndarray] = [np.zeros_like(p.data) for p in parameters]
        self._v: List[np.ndarray] = [np.zeros_like(p.data) for p in parameters]
        self._t: int = 0

    def step(self) -> None:
        """
        Tek optimizasyon adımı.
        """
        self._t += 1
        for idx, p in enumerate(self.parameters):
            grad = p.grad
            if grad is None:
                continue
            # 1. moment (eğik ortalama)
            self._m[idx] = self.beta1 * self._m[idx] + (1 - self.beta1) * grad
            # 2. moment (eğik ortalama kare)
            self._v[idx] = self.beta2 * self._v[idx] + (1 - self.beta2) * (grad ** 2)
            # Bias düzeltme
            m_hat = self._m[idx] / (1 - self.beta1 ** self._t)
            v_hat = self._v[idx] / (1 - self.beta2 ** self._t)
            # Parametre güncellemesi
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data = p.data - update

    def zero_grad(self) -> None:
        """
        Tüm parametrelerin gradyanlarını temizler.
        """
        for p in self.parameters:
            p.grad = None

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, step={self._t})"
        )