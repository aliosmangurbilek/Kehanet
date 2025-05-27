# core/losses/cross_entropy.py
"""
Softmax + Negatif Log-Likelihood (Cross‐Entropy) Loss
=====================================================
"""
from __future__ import annotations
import numpy as np
from core.tensor import Tensor


def cross_entropy(
    pred: Tensor,
    target: Tensor
) -> Tensor:
    """
    Cross-Entropy loss using softmax and NLL.

    Parameters
    ----------
    pred : Tensor
        Modelin ham çıktıları (logits), şekil (N, C).
    target : Tensor
        One-hot kodlanmış gerçek etiketler, şekil (N, C).

    Returns
    -------
    Tensor
        Tek değerli skaler kayıp (requires_grad=True).
    """
    logits = pred.data  # (N, C)
    labels = target.data  # (N, C)
    N, C = logits.shape

    # Numerik stabilite için kaydırma
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    probs = exp_shifted / sum_exp  # (N, C)
    log_probs = shifted - np.log(sum_exp)

    # Örnek başına kayıp
    losses = -np.sum(labels * log_probs, axis=1)  # (N,)
    loss_value = float(np.mean(losses))  # Python scalar

    # Oluşturulan kayıp tensörü
    loss = Tensor(loss_value, requires_grad=True)

    # -------- Autograd geri yayılım --------
    def _backward() -> None:
        if pred.requires_grad:
            grad_pred = (probs - labels) / N  # (N, C)
            pred.grad = (
                pred.grad if pred.grad is not None else np.zeros_like(pred.data)
            ) + grad_pred

    loss._backward = _backward
    loss._prev = {pred}
    return loss
