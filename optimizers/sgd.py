# kehanet/optimizers/sgd.py
"""
Stokastik Gradyan İnişi (SGD)
=============================

Kullanım:
    optimizer = SGD(model.parameters(), lr=0.01)
    ...
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
"""

from __future__ import annotations
from typing import Iterable, List
from core.tensor import Tensor


class SGD:
    def __init__(self, params: Iterable[Tensor], lr: float = 0.01):
        self.params: List[Tensor] = list(params)
        self.lr = lr

    def step(self):
        """Her parametreyi p = p - lr * p.grad ile günceller."""
        for p in self.params:
            if p.grad is None:
                continue
            p.data -= self.lr * p.grad

    def zero_grad(self):
        """Parametre gradyanlarını sıfırlar."""
        for p in self.params:
            p.grad = None
