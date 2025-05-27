# core/sequential.py
from __future__ import annotations
from typing import List, Protocol
from core.tensor import Tensor

class Layer(Protocol):
    """
    Katmanların uyması gereken arayüz (protocol).
    """
    training: bool
    def __call__(self, x: Tensor) -> Tensor: ...
    def parameters(self) -> List[Tensor]: ...

class SimpleSequential:
    """
    Basit bir Sequential model sınıfı.
    train() / eval() modunu her katmana yansıtır.
    """

    def __init__(self, layers: List[Layer]):
        self._layers: List[Layer] = list(layers)
        self.training: bool = False

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._layers:
            layer.training = self.training
            x = layer(x)
        return x

    __call__ = forward

    def train(self) -> SimpleSequential:
        """Modeli eğitim moduna alır ve referansı döner."""
        self.training = True
        return self

    def eval(self) -> SimpleSequential:
        """Modeli değerlendirme moduna alır ve referansı döner."""
        self.training = False
        return self

    def parameters(self) -> List[Tensor]:
        """Tüm katmanlardan toplanan parametreleri listeler."""
        params: List[Tensor] = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params

    def __len__(self) -> int:
        """Katman sayısını döner."""
        return len(self._layers)

    def __getitem__(self, idx: int) -> Layer:
        """İndekse göre katmana erişim sağlar."""
        return self._layers[idx]

    def __repr__(self) -> str:
        """Modeli ve katmanları özetleyen temsil string'i."""
        layer_reprs = ',\n  '.join(repr(l) for l in self._layers)
        return f"{self.__class__.__name__}([\n  {layer_reprs}\n])"
