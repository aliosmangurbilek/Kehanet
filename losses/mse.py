# kehanet/losses/mse.py
"""
Ortalama Kare Hatası (MSE) Loss
===============================

Kullanım:
    from kehanet.losses.mse import mse_loss
    loss = mse_loss(pred, target)   # loss: Tensor
    loss.backward()                 # gradyanlar hesaplanır
"""

from __future__ import annotations
import numpy as np
from core.tensor import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Ortalama Kare Hatası (Mean Squared Error).

    Parameters
    ----------
    pred : Tensor
        Model çıktıları.
    target : Tensor
        Gerçek değerler.

    Returns
    -------
    Tensor
        Skaler kayıp değeri (requires_grad=True).
    """
    assert pred.data.shape == target.data.shape, "Şekiller uyuşmuyor"
    n = pred.data.size
    diff = pred - target        # (pred - target)
    sq = Tensor(diff.data ** 2) # eleman bazlı kare; diff Tensor'ı sayesinde grad bağlantısı var
    loss_value = np.sum(sq.data) / n
    loss = Tensor(loss_value, requires_grad=True)

    # -------- autograd bağlantısı -------- #
    def _backward():
        if pred.requires_grad:
            grad_pred = (2.0 / n) * diff.data          # dL/dpred
            pred.grad = (pred.grad or 0) + grad_pred
        # target için gradyan hesaplamıyoruz (genelde requires_grad=False)
    loss._backward = _backward
    loss._prev = {pred}
    return loss