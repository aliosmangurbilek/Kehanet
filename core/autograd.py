# kehanet/core/autograd.py
"""
Basit otomatik türev motoru.

Her Tensor, _backward fonksiyonu ve _prev kümesi (bağlı parent tensorlar)
barındırır.  Burada:
    * topological_sort  : Grafı DAG sıralamasına sokar.
    * backward          : Kökten geriye yayılımı başlatır.
    * zero_grad         : Gradyanları temizler.
"""

from __future__ import annotations
import numpy as np
from typing import List, Set, Callable
from core.tensor import Tensor


def topological_sort(tensor: Tensor) -> List[Tensor]:
    """Çıktı tensöründen başlayarak DAG sırası oluşturur."""
    seen: Set[Tensor] = set()
    order: List[Tensor] = []

    def build(t: Tensor):
        if t not in seen:
            seen.add(t)
            for child in t._prev:
                build(child)
            order.append(t)

    build(tensor)
    return order


def backward(tensor: Tensor, grad: np.ndarray | None = None) -> None:
    """
    Geri yayılımı tensor.backward(...) ile aynen çalıştırır
    ancak merkezi tek fonksiyon olarak tutar.
    """
    if not tensor.requires_grad:
        raise RuntimeError("Gradyan gerektirmeyen tensor için backward çağrıldı.")

    if grad is None:
        # Çıktı skaler ise otomatik olarak 1 grad başlat
        if tensor.data.size != 1:
            raise RuntimeError(
                "Grad parametresiz backward, yalnızca skaler çıktı için geçerlidir."
            )
        grad = np.ones_like(tensor.data, dtype=np.float32)

    # Topolojik sıra
    topo = topological_sort(tensor)
    tensor.grad = grad

    # Bu sırayı tersten gez, her düğümde _backward çağır
    for t in reversed(topo):
        t._backward()


def zero_grad(tensor: Tensor) -> None:
    """
    Graf içindeki tüm tensörlerin grad alanlarını None yapar.
    """
    for t in topological_sort(tensor):
        t.grad = None


# -------------------------- Aktivasyon İşlevleri -------------------------- #
def relu(x: Tensor) -> Tensor:
    out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            x.grad = (x.grad or 0) + out.grad * (x.data > 0)

    out._backward = _backward
    out._prev = {x}
    return out


def sigmoid(x: Tensor) -> Tensor:
    s = 1 / (1 + np.exp(-x.data))
    out = Tensor(s, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            x.grad = (x.grad or 0) + out.grad * s * (1 - s)

    out._backward = _backward
    out._prev = {x}
    return out


def softmax(x: Tensor) -> Tensor:
    """
    Softmax aktivasyon fonksiyonu.
    Boyutlar arası (genellikle son boyut) normalizasyon sağlar.

    Args:
        x: Girdi tensörü, genellikle son boyut üzerinde softmax uygulanır

    Returns:
        Softmax uygulanmış tensör
    """
    # Numerik stabilite için max çıkartma
    shifted_x = x.data - np.max(x.data, axis=-1, keepdims=True)
    exp_x = np.exp(shifted_x)
    s = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    out = Tensor(s, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            # Softmax'in karmaşık gradyan hesabı:
            # Jacobian: ∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
            # Bu işlem batch işlemek için uygundur
            batch_size = out.data.shape[0]
            n_classes = out.data.shape[-1]

            # Başlangıçta gradyan initialize et
            dx = np.zeros_like(x.data)

            # Her örnek için döngü
            for i in range(batch_size):
                # Örnek için softmax çıktısı
                sm = out.data[i]
                # Örnek için gelen gradyan
                dout = out.grad[i]

                # S * (I - S^T)
                # S: softmax çıktısı (n_classes,)
                # I: birim matris (n_classes, n_classes)
                # S^T: softmax çıktısının transpozu (n_classes,)

                # Jacobian matrisi hesaplama
                jacobian = np.diag(sm) - np.outer(sm, sm)

                # Gradyan ile Jacobian'ı çarp
                dx[i] = dout @ jacobian

            # Gradyanı mevcut gradyan ile topla
            x.grad = (x.grad or 0) + dx

    out._backward = _backward
    out._prev = {x}
    return out
