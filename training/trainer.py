# kehanet/training/trainer.py
"""
Trainer
=======

Bu sınıf, model eğitimi ve değerlendirme işlemlerini yönetir.

Kullanım:
    from kehanet.training.trainer import Trainer

    trainer = Trainer(
        model=sequential_model,
        loss_fn=mse_loss,
        optimizer=sgd_optimizer,
        train_loader=train_loader,
        test_loader=test_loader
    )
    trainer.train(epochs=10)
    trainer.evaluate()
"""

from typing import Callable, Optional
import numpy as np
from core.tensor import Tensor

class Trainer:
    def __init__(
        self,
        model,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer,
        train_loader,
        test_loader: Optional = None
    ):
        """
        model        : Katmanlardan oluşan model (sequential veya benzeri)
        loss_fn      : Kayıp fonksiyonu (pred, target) -> Tensor
        optimizer    : SGD veya Adam gibi optimizer nesnesi
        train_loader : (X_batch, y_batch) üreten bir iterable
        test_loader  : (X_batch, y_batch) üreten isteğe bağlı iterable
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, epochs: int = 1) -> None:
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            # Eğitim döngüsü
            for X_batch, y_batch in self.train_loader:
                # Tensor'e çevir
                X = Tensor(X_batch, requires_grad=False)
                y = Tensor(y_batch, requires_grad=False)

                # Gradients sıfırla
                for p in self.model.parameters():
                    p.grad = None

                # İleri yayılım
                preds = self.model(X)
                loss = self.loss_fn(preds, y)

                # Geri yayılım
                loss.backward()

                # Ağırlık güncelleme
                self.optimizer.step()

                # Gradients temizle
                self.optimizer.zero_grad()

                epoch_loss += loss.data
                n_batches += 1

            avg_loss = epoch_loss / n_batches if n_batches else float('nan')
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    def evaluate(self) -> None:
        if self.test_loader is None:
            print("Test loader tanımlı değil.")
            return

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        n_batches = 0

        for X_batch, y_batch in self.test_loader:
            X = Tensor(X_batch, requires_grad=False)
            y = Tensor(y_batch, requires_grad=False)
            preds = self.model(X)
            loss = self.loss_fn(preds, y)
            total_loss += loss.data

            # Doğruluk hesaplama (one-hot etiket varsayımıyla)
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