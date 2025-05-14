# kehanet/core/tensor.py
from __future__ import annotations
import numpy as np
from typing import Callable

class Tensor:
    """
    Basit Tensor sınıfı. NumPy dizilerini sarar,
    autograd için altyapı sağlar.
    """
    def __init__(self, data, *, requires_grad: bool = False):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Tensor] = set()

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    # ---------- Temel İşlemler ----------

    def __add__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=(self.requires_grad or other.requires_grad)
        )
        def _backward():
            grad = out.grad
            # self grad
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) + grad
            # other grad with broadcast handling
            if other.requires_grad:
                other_grad = grad
                # handle broadcasting dimensions
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

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) - out.grad
        out._backward = _backward
        out._prev = {self}
        return out

    def __sub__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
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

    # ---------- Geri Yayılım ----------

    def backward(self, grad: np.ndarray | None = None):
        """
        Geri yayılım. Eğer grad belirtilmezse skaler Tensor için 1 ile başlar.
        """
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