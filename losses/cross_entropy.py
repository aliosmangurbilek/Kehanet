# kehanet/losses/cross_entropy.py
"""
Softmax + Negatif Log-Likelihood (Cross‐Entropy) Loss
=====================================================

Kullanım:
    from kehanet.losses.cross_entropy import cross_entropy
    loss = cross_entropy(logits, targets)   # loss: Tensor
    loss.backward()                         # gradyanlar hesaplanır
"""

from __future__ import annotations
import numpy as np
from core.tensor import Tensor

def cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    """
    Cross‐Entropy loss for classification.

    Parameters
    ----------
    pred : Tensor
        Modelin ham çıktıları (logits), şekil (N, C).
    target : Tensor
        One‐hot kodlanmış gerçek etiketler, şekil (N, C).

    Returns
    -------
    Tensor
        Skaler kayıp değeri (requires_grad=True).
    """
    logits = pred.data
    labels = target.data
    N, C = logits.shape

    # numerical stability için max logit'i çıkar
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    log_probs = logits - max_logits - np.log(sum_exp)  # (N, C)

    # örnek başına kayıp
    losses = -np.sum(labels * log_probs, axis=1)  # (N,)
    loss_value = np.mean(losses)  # skaler

    loss = Tensor(loss_value, requires_grad=True)

    # -------- autograd bağlantısı -------- #
    def _backward():
        if pred.requires_grad:
            # softmax grads: (exp/sum) - one_hot, normalize by N
            probs = exp_logits / sum_exp  # (N, C)
            grad_pred = (probs - labels) / N  # (N, C)
            pred.grad = (pred.grad or 0) + grad_pred

    loss._backward = _backward
    loss._prev = {pred}
    return loss
