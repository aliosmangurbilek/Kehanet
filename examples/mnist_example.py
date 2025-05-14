# examples/mnist_example.py
"""
MNIST veri seti ile Kehanet kütüphanesini kullanarak basit bir eğitim örneği.
"""
import numpy as np

# Package imports with 'kehanet' prefix
from core.tensor import Tensor
from layers.dense import Dense
from optimizers.sgd import SGD
from losses.cross_entropy import cross_entropy
from data.dataloader import load_mnist_local, DataLoader
from training.trainer import Trainer
from core.autograd import relu

# 1. Veri hazırlığı: datasets/mnist içinde bulunan dosyaları kullan
train_ds, test_ds = load_mnist_local(
    folder='/home/ali/PycharmProjects/Kehanet/datasets/mnist', normalize=True, one_hot=True
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

# 2. Model tanımı: 784 -> 128 -> 10
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
    Dense(784, 128, activation=relu, bias=True),
    Dense(128, 10, activation=None, bias=True)
])

# 3. Optimizer ve Trainer oluşturma
optimizer = SGD(model.parameters(), lr=0.01)
trainer = Trainer(
    model=model,
    loss_fn=cross_entropy,
    optimizer=optimizer,
    train_loader=train_loader,
    test_loader=test_loader
)


# 4. Eğitim
if __name__ == '__main__':
    epochs = 5
    trainer.train(epochs=epochs)
    trainer.evaluate()