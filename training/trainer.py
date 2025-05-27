# core/training/trainer.py
"""
Trainer
=======

Model eğitimi ve değerlendirmesini yöneten yardımcı sınıf.
"""
from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np
from core.tensor import Tensor

class Trainer:
    """
    Trainer sınıfı, eğitim ve test döngülerini içerir.

    Attributes
    ----------
    model : Any
        Eğitim yapılacak model (SimpleSequential gibi).
    loss_fn : Callable[[Tensor, Tensor], Tensor]
        Kayıp fonksiyonu.
    optimizer : Any
        Optimizer nesnesi (SGD, Adam vb.).
    train_loader : Iterable[Tuple[np.ndarray, np.ndarray]]
        Eğitim verisi loader.
    test_loader : Optional[Iterable[Tuple[np.ndarray, np.ndarray]]]
        Test verisi loader.
    """
    def __init__(
        self,
        model,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer,
        train_loader,
        test_loader: Optional = None
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, epochs: int = 1) -> None:
        """
        Modelin eğitim döngüsünü yürütür.
        """
        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in self.train_loader:
                X = Tensor(X_batch, requires_grad=False)
                y = Tensor(y_batch, requires_grad=False)
                # Gradients sıfırla
                self.optimizer.zero_grad()
                # İleri yayılım
                preds = self.model(X)
                loss = self.loss_fn(preds, y)
                # Geri yayılım
                loss.backward()
                # Parametre güncelleme
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / n_batches if n_batches else float('nan')
            print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.4f}")

    def evaluate(self) -> Tuple[float, float]:
        """
        Modeli test verisi üzerinde değerlendirir.

        Returns
        -------
        Tuple[float, float]
            (ortalama kayıp, doğruluk)
        """
        if self.test_loader is None:
            print("Test loader tanımlı değil.")
            return float('nan'), float('nan')

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        n_batches = 0

        for X_batch, y_batch in self.test_loader:
            X = Tensor(X_batch, requires_grad=False)
            y = Tensor(y_batch, requires_grad=False)
            preds = self.model(X)
            loss = self.loss_fn(preds, y)
            total_loss += loss.item()

            # Doğruluk
            pred_labels = np.argmax(preds.data, axis=1)
            if y_batch.ndim > 1:
                true_labels = np.argmax(y_batch, axis=1)
            else:
                true_labels = y_batch.astype(int)
            total_correct += np.sum(pred_labels == true_labels)
            total_samples += X_batch.shape[0]
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches else float('nan')
        accuracy = total_correct / total_samples if total_samples else float('nan')
        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy