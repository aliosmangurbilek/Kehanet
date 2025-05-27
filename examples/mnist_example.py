# examples/mnist_example.py
"""
MNIST eğitimi için örnek script.
Argümanlarla esnek eğitim konfigürasyonu ve opsiyonel augmentasyon desteği içerir.
"""
import argparse
import os
import pickle
import numpy as np
import cv2

from core.tensor import Tensor
from core.autograd import relu
from core.sequential import SimpleSequential
from data.dataloader import load_mnist_local, DataLoader
from layers.dense import Dense
from optimizers.adam import Adam
from losses.cross_entropy import cross_entropy
from training.trainer import Trainer

# --- 1) Augmentasyon fonksiyonu ---
def augment_mnist(batch: np.ndarray) -> np.ndarray:
    """
    Her görüntüyü küçük rastgele kaydır ve döndür.
    """
    imgs = batch.reshape(-1, 28, 28)
    out = []
    for img in imgs:
        # rastgele kaydırma
        tx, ty = np.random.randint(-2, 3), np.random.randint(-2, 3)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(img, M, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # rastgele döndürme
        angle = np.random.uniform(-5, 5)
        R = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        rotated = cv2.warpAffine(shifted, R, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out.append(rotated.flatten())
    return np.stack(out).astype(np.float32)

# --- 2) Argument parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="MNIST örnek eğitim scripti.")
    parser.add_argument("--data-dir", default="/home/ali/PycharmProjects/Kehanet/datasets/mnist/", help="MNIST veri dizini")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--augment", action="store_true", help="Eğitime augmentasyon ekle")
    parser.add_argument("--save-path", default="/home/ali/PycharmProjects/Kehanet/models/trained_mnist_model.pkl", help="Kaydedilecek model dosyası")
    return parser.parse_args()

# --- 3) Ana fonksiyon ---
def main():
    args = parse_args()
    np.random.seed(42)

    # Veriyi yükle
    train_ds, test_ds = load_mnist_local(
        folder=args.data_dir, normalize=True, one_hot=True
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model tanımı
    model = SimpleSequential([
        Dense(784, 128, activation=relu, bias=True, dropout=0.2),
        Dense(128, 64,  activation=relu, bias=True, dropout=0.2),
        Dense(64, 32,   activation=relu, bias=True, dropout=0.2),
        Dense(32, 10,   activation=None, bias=True),  # logits
    ])

    # Optimizer ve Trainer
    optimizer = Adam(model.parameters(), lr=args.lr)
    trainer   = Trainer(
        model=model,
        loss_fn=cross_entropy,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader
    )

    # Eğitim döngüsü
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss, count = 0.0, 0
        for Xb, yb in train_loader:
            X = Tensor(
                augment_mnist(Xb) if args.augment else Xb,
                requires_grad=False
            )
            y = Tensor(yb, requires_grad=False)

            # gradyan temizle
            for p in model.parameters():
                p.grad = None

            preds = model(X)
            loss = cross_entropy(preds, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

    # Değerlendirme
    model.eval()
    trainer.evaluate()

    # Modeli kaydet
    save_dir = os.path.dirname(args.save_path)
    if save_dir:  # dizin kısmı boş değilse oluştursun
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model kaydedildi: {args.save_path}")

if __name__ == "__main__":
    main()
