# core/tensor.py
from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Union, Set, Tuple

class Tensor:
    """
    Basit Tensor sınıfı. NumPy dizilerini sarar,
    autograd için altyapı sağlar.
    """
    def __init__(
        self,
        data: Union[float, list, np.ndarray],
        *,
        requires_grad: bool = False
    ):
        self.data: np.ndarray = np.asarray(data, dtype=np.float32)
        self.requires_grad: bool = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward: Callable[[], None] = self._default_backward
        self._prev: Set[Tensor] = set()

    def _default_backward(self) -> None:
        """Gradient gerektirmeyen işlemler için no-op."""
        pass

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def numpy(self) -> np.ndarray:
        """Altındaki NumPy dizisini döner."""
        return self.data

    def item(self) -> float:
        """Sadece tek bir eleman varsa Python scalara çevirir."""
        return self.data.item()

    def detach(self) -> Tensor:
        """Yeni bir requires_grad=False Tensor döner."""
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self) -> None:
        """Graf kapsamındaki tüm grad alanlarını sıfırlar."""
        for t in self._topological_sort():
            t.grad = None

    # ---------- Temel Operatörler ----------

    def __add__(self, other: Tensor | float) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=(self.requires_grad or other.requires_grad)
        )
        def _backward():
            grad = out.grad
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) + grad
            if other.requires_grad:
                other_grad = grad
                if other.data.shape != grad.shape:
                    axes = tuple(
                        i for i,(s_o, s_g) in enumerate(zip(other.data.shape, grad.shape))
                        if s_o == 1 and s_g > 1
                    )
                    if axes:
                        other_grad = np.sum(grad, axis=axes, keepdims=True)
                other.grad = (other.grad or np.zeros_like(other.data)) + other_grad
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other: Tensor | float) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=(self.requires_grad or other.requires_grad)
        )
        def _backward():
            grad = out.grad
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) + other.data * grad
            if other.requires_grad:
                other.grad = (other.grad or np.zeros_like(other.data)) + self.data * grad
        out._backward = _backward
        out._prev = {self, other}
        return out
    __rmul__ = __mul__

    def __neg__(self) -> Tensor:
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) - out.grad
        out._backward = _backward
        out._prev = {self}
        return out

    def __sub__(self, other: Tensor | float) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __matmul__(self, other: Tensor | float) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=(self.requires_grad or other.requires_grad)
        )
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) + out.grad @ other.data.T
            if other.requires_grad:
                other.grad = (other.grad or np.zeros_like(other.data)) + self.data.T @ out.grad
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __truediv__(self, other: Tensor | float) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=(self.requires_grad or other.requires_grad)
        )
        def _backward():
            grad = out.grad
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) + grad / other.data
            if other.requires_grad:
                other_grad = -self.data * grad / (other.data ** 2)
                other.grad = (other.grad or np.zeros_like(other.data)) + other_grad
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __rtruediv__(self, other: Tensor | float) -> Tensor:
        return Tensor(other) / self

    def __pow__(self, exponent: float) -> Tensor:
        out = Tensor(self.data ** exponent, requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) + exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        out._prev = {self}
        return out

    # ---------- Redüksiyonlar ----------

    def sum(self, axis=None, keepdims=False) -> Tensor:
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis)
                self.grad = (self.grad or np.zeros_like(self.data)) + np.broadcast_to(grad, self.data.shape)
        out._backward = _backward
        out._prev = {self}
        return out

    def reshape(self, *shape: int) -> Tensor:
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) + out.grad.reshape(self.data.shape)
        out._backward = _backward
        out._prev = {self}
        return out

    # ---------- Geri Yayılım ----------

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("backward() çağrılan Tensor requires_grad=False")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad parametresiz backward yalnızca skaler Tensor için geçerli."
                )
            grad = np.ones_like(self.data, dtype=np.float32)
        self.grad = grad
        for t in reversed(self._topological_sort()):
            t._backward()

    def _topological_sort(self) -> list[Tensor]:
        seen, order = set(), []
        def build(t: Tensor):
            if t not in seen:
                seen.add(t)
                for prev in t._prev:
                    build(prev)
                order.append(t)
        build(self)
        return order