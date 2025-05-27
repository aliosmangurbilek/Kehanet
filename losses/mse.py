# core/losses/mse.py
"""
Ortalama Kare Hatası (MSE) Loss
===============================
"""
from __future__ import annotations
import numpy as np
from core.tensor import Tensor


def mse_loss(
    pred: Tensor,
    target: Tensor
) -> Tensor:
    """
    Ortalama Kare Hatası (Mean Squared Error).

    Parameters
    ----------
    pred : Tensor
        Model çıktıları.
    target : Tensor
        Gerçek değerler (aynı şekil).

    Returns
    -------
    Tensor
        Tek değerli skaler kayıp (requires_grad=True).
    """
    # Şekiller eşit olmalı
    if pred.data.shape != target.data.shape:
        raise ValueError(
            f"pred ve target şekilleri uyuşmuyor: {pred.data.shape} vs {target.data.shape}"
        )
    # Fark ve kareleri hesapla
    diff = pred - target  # Tensor
    # Eleman bazlı kare
    sq = diff * diff      # Tensor
    # Ortalama kare hatası
    loss_value = float(np.mean(sq.data))
    loss = Tensor(loss_value, requires_grad=True)

    # -------- Autograd geri yayılım --------
    def _backward() -> None:
        if pred.requires_grad:
            # dL/dpred = 2*(pred - target) / n_elem
            n = pred.data.size
            grad_pred = (2.0 / n) * diff.data
            pred.grad = (
                pred.grad if pred.grad is not None else np.zeros_like(pred.data)
            ) + grad_pred
    loss._backward = _backward
    loss._prev = {pred}
    return loss
