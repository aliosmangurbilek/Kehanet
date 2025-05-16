# examples/mnist_example.py
"""
MNIST veri seti ile Kehanet kütüphanesini kullanarak basit bir eğitim örneği.
"""
import numpy as np

# Package imports with 'kehanet' prefix
from core.tensor import Tensor
from layers.dense import Dense
from optimizers.adam import Adam  # Change from SGD to Adam
from losses.cross_entropy import cross_entropy
from data.dataloader import load_mnist_local, DataLoader
from training.trainer import Trainer
from core.autograd import relu, sigmoid, softmax
import pickle  # Add import for saving the model
from core.sequential import SimpleSequential  # Updated import

# 1. Veri hazırlığı: datasets/mnist içinde bulunan dosyaları kullan
train_ds, test_ds = load_mnist_local(
    folder='/home/ali/PycharmProjects/Kehanet/datasets/mnist', normalize=True, one_hot=True
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

# 2. Model tanımı: 784 -> 256 -> 128 -> 10
class SimpleSequential:
    def __init__(self, layers):
        self._layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self._layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params

model = SimpleSequential([
    Dense(784, 128, activation=relu, bias=True),  # Daha büyük bir ara katman
    Dense(128, 64, activation=sigmoid, bias=True),
    Dense(64, 32, activation=sigmoid, bias=True),
    Dense(32, 10, activation=softmax, bias=True)
])

# 3. Optimizer ve Trainer oluşturma
optimizer = Adam(model.parameters(), lr=0.001)  # Changed from SGD to Adam with default parameters
trainer = Trainer(
    model=model,
    loss_fn=cross_entropy,
    optimizer=optimizer,
    train_loader=train_loader,
    test_loader=test_loader
)


# 4. Eğitim
if __name__ == '__main__':
    epochs = 50
    trainer.train(epochs=epochs)
    trainer.evaluate()

    # Save the trained model
    model_path = '/home/ali/PycharmProjects/Kehanet/models/trained_mnist_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Trained model saved to {model_path}")
