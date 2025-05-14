# kehanet/layers/base.py
"""
Katman Temel Sınıfı
===================

Tüm katmanlar bu sınıftan türetilir. Ortak işlevler:
    * __call__   : forward delegasyonu
    * parameters : katmana ait öğrenilebilir tensörleri döndürür
    * zero_grad  : parametre gradyanlarını sıfırlar
"""

from __future__ import annotations
from typing import List
from core.tensor import Tensor


class Layer:
    def __init__(self):
        self._params: List[Tensor] = []

    # ----------------------- Geçersiz kılınacak kısım ---------------------- #
    def forward(self, x: Tensor) -> Tensor:
        """İleri yayılım – alt sınıfta uygulanmalı."""
        raise NotImplementedError

    # --------------------------- Ortak yardımcılar ------------------------- #
    def __call__(self, x: Tensor) -> Tensor:
        """Katmanı fonksiyon gibi çağırma kolaylığı."""
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        """Öğrenilebilir tensörleri (ağırlık, bias) döndürür."""
        return self._params

    def zero_grad(self):
        """Parametrelere ait gradyanları temizler."""
        for p in self._params:
            p.grad = None
